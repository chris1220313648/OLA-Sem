#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate language_image text files for an existing RobotWin dataset.

This script scans:
  target_root/{clean,randomized}/{task}/{input_dir_name}/*.mp4
and writes:
  target_root/{clean,randomized}/{task}/language_image/*.txt

Each output text file contains one line per timestep. Line t summarizes scene
changes from a sliding window:
  frames[t : min(t + window_size, T)]

Per line generation uses 3 sampled frames from the window:
  - first frame
  - middle frame
  - last frame

Text generation is done with a Qwen-VL model and constrained to:
  - focus on dynamic changes
  - include change/edit regions
  - keep one short line for supervision stability
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from decord import VideoReader, cpu
from PIL import Image
from qwen_vl_utils import process_vision_info
from tqdm.auto import tqdm
from transformers import AutoProcessor
from transformers import Qwen3VLForConditionalGeneration

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


SYSTEM_PROMPT = (
    "You are a robotics scene-change describer. "
    "Describe how the objects and robot arm will change across frames, including approximate directions (for example: left, right, up, down, forward, backward). "
    # "Use future tense grammar (for example, use 'will'). "
    "Describe only dynamic changes and avoid static background description."
)


def _natural_key(path_or_name: str) -> list[object]:
    return [int(tok) if tok.isdigit() else tok.lower() for tok in re.split(r"(\d+)", path_or_name)]


def _iter_task_dirs(subset_dir: Path) -> Iterable[Path]:
    task_dirs = [p for p in subset_dir.iterdir() if p.is_dir()]
    for task_dir in sorted(task_dirs, key=lambda p: _natural_key(p.name)):
        yield task_dir


def _sanitize_line(text: str) -> str:
    line = " ".join(text.strip().split())
    return line if line else "No significant scene change; editable region: none."


def _sample_three_indices(start: int, end: int) -> list[int]:
    """Sample first/middle/last indices from [start, end)."""
    if end <= start:
        return [start]
    first = start
    last = end - 1
    middle = start + (end - start - 1) // 2

    sampled: list[int] = []
    for idx in (first, middle, last):
        if not sampled or idx != sampled[-1]:
            sampled.append(idx)
    return sampled


def _build_timestep_specs(timesteps: int, window_size: int) -> list[tuple[int, int, list[int]]]:
    """Build (start_idx, end_idx, sampled_indices) for each timestep."""
    specs: list[tuple[int, int, list[int]]] = []
    for start in range(timesteps):
        end = min(start + window_size, timesteps)
        sampled = _sample_three_indices(start, end)
        specs.append((start, end, sampled))
    return specs


def _collect_unique_indices(step_specs: list[tuple[int, int, list[int]]]) -> list[int]:
    """Collect sorted unique frame indices needed by a micro-batch."""
    idx_set: set[int] = set()
    for _, _, sampled in step_specs:
        idx_set.update(sampled)
    return sorted(idx_set)


def _torch_dtype_from_string(dtype: str) -> torch.dtype | str:
    mapping: dict[str, torch.dtype | str] = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if dtype not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype}")
    return mapping[dtype]


class QwenVLSceneChangeGenerator:
    """Wrapper for Qwen-VL inference from sampled frames."""

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda:0",
        dtype: str = "auto",
        max_new_tokens: int = 96,
        temperature: float = 0.0,
        top_p: float = 0.9,
    ) -> None:
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.top_p = float(top_p)

        torch_dtype = _torch_dtype_from_string(dtype)

        logger.info(
            "Loading Qwen-VL model: checkpoint=%s device=%s dtype=%s",
            checkpoint_path,
            device,
            dtype,
        )
        self.processor = AutoProcessor.from_pretrained(checkpoint_path, trust_remote_code=True)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            checkpoint_path,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )
        self.model = self.model.to(device)
        self.model.eval()

    def _build_user_prompt(self, task_name: str, start_idx: int, end_idx: int) -> str:
        return (
            f"Task: {task_name}. "
            f"Frames from timestep {start_idx} to {max(start_idx, end_idx - 1)} are provided. "
            "Describe scene/object changes only (not static background). "
            "State the key change region or editable region explicitly. "
            "Output one short sentence."
        )

    def _build_messages(
        self,
        task_name: str,
        images: list[Image.Image],
        start_idx: int,
        end_idx: int,
    ) -> list[dict[str, object]]:
        user_content: list[dict[str, object]] = []
        for image in images:
            user_content.append({"type": "image", "image": image})
        user_content.append(
            {
                "type": "text",
                "text": self._build_user_prompt(task_name=task_name, start_idx=start_idx, end_idx=end_idx),
            }
        )
        return [
            {
                "role": "system",
                "content": [{"type": "text", "text": SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": user_content,
            },
        ]

    @torch.inference_mode()
    def generate_lines_batch(self, batch_items: list[dict[str, object]]) -> list[str]:
        if not batch_items:
            return []

        messages_batch: list[list[dict[str, object]]] = []
        texts: list[str] = []
        for item in batch_items:
            messages = self._build_messages(
                task_name=str(item["task_name"]),
                images=item["images"],  # type: ignore[arg-type]
                start_idx=int(item["start_idx"]),
                end_idx=int(item["end_idx"]),
            )
            messages_batch.append(messages)
            texts.append(self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

        image_inputs, video_inputs = process_vision_info(messages_batch)
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        generate_kwargs: dict[str, object] = {
            "max_new_tokens": self.max_new_tokens,
        }
        if self.temperature > 0:
            generate_kwargs["do_sample"] = True
            generate_kwargs["temperature"] = self.temperature
            generate_kwargs["top_p"] = self.top_p
        else:
            generate_kwargs["do_sample"] = False

        generated_ids = self.model.generate(**inputs, **generate_kwargs)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        outputs = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return [_sanitize_line(text) for text in outputs]

    @torch.inference_mode()
    def generate_line(
        self,
        task_name: str,
        images: list[Image.Image],
        start_idx: int,
        end_idx: int,
    ) -> str:
        return self.generate_lines_batch(
            [
                {
                    "task_name": task_name,
                    "images": images,
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                }
            ]
        )[0]


@dataclass
class BackfillStats:
    tasks_scanned: int = 0
    episodes_total: int = 0
    episodes_generated: int = 0
    episodes_skipped_existing: int = 0
    episodes_failed: int = 0
    generated_lines: int = 0

    def merge(self, other: "BackfillStats") -> None:
        self.tasks_scanned += other.tasks_scanned
        self.episodes_total += other.episodes_total
        self.episodes_generated += other.episodes_generated
        self.episodes_skipped_existing += other.episodes_skipped_existing
        self.episodes_failed += other.episodes_failed
        self.generated_lines += other.generated_lines


class RobotWinLanguageImageBackfill:
    """Backfill language_image text files under RobotWin target_root."""

    def __init__(
        self,
        target_root: Path,
        subsets: list[str],
        vlm_checkpoint_path: str,
        window_size: int = 16,
        input_dir_name: str = "videos",
        output_dir_name: str = "language_image",
        device: str = "cuda:0",
        dtype: str = "auto",
        max_new_tokens: int = 96,
        temperature: float = 0.0,
        top_p: float = 0.9,
        gen_batch_size: int = 1,
        video_num_threads: int = 2,
        show_progress: bool = True,
        overwrite: bool = False,
    ) -> None:
        self.target_root = target_root
        self.subsets = subsets
        self.window_size = int(window_size)
        self.input_dir_name = input_dir_name
        self.output_dir_name = output_dir_name
        self.gen_batch_size = int(gen_batch_size)
        self.video_num_threads = int(video_num_threads)
        self.show_progress = show_progress
        self.overwrite = overwrite

        self._validate()

        self.generator = QwenVLSceneChangeGenerator(
            checkpoint_path=vlm_checkpoint_path,
            device=device,
            dtype=dtype,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

    def _validate(self) -> None:
        if not self.target_root.exists():
            raise FileNotFoundError(f"target_root not found: {self.target_root}")
        if not self.subsets:
            raise ValueError("subsets is empty")
        if self.window_size <= 0:
            raise ValueError(f"window_size must be > 0, got {self.window_size}")
        if self.gen_batch_size <= 0:
            raise ValueError(f"gen_batch_size must be > 0, got {self.gen_batch_size}")
        if self.video_num_threads <= 0:
            raise ValueError(f"video_num_threads must be > 0, got {self.video_num_threads}")
        if not self.input_dir_name.strip():
            raise ValueError("input_dir_name must be non-empty")
        if not self.output_dir_name.strip():
            raise ValueError("output_dir_name must be non-empty")

    def _load_frame_map(self, video_reader: VideoReader, indices: list[int]) -> dict[int, Image.Image]:
        if not indices:
            return {}
        batch_np = video_reader.get_batch(indices).asnumpy()  # [N, H, W, C], RGB uint8
        frame_map: dict[int, Image.Image] = {}
        for pos, idx in enumerate(indices):
            frame_map[idx] = Image.fromarray(batch_np[pos], mode="RGB")
        return frame_map

    def _process_episode(self, task_name: str, video_path: Path, out_path: Path) -> tuple[bool, int]:
        vr = VideoReader(str(video_path), ctx=cpu(0), num_threads=self.video_num_threads)
        timesteps = len(vr)
        if timesteps <= 0:
            raise ValueError(f"Video has no frames: {video_path}")

        step_specs = _build_timestep_specs(timesteps=timesteps, window_size=self.window_size)
        lines: list[str] = [""] * timesteps

        frame_pbar = None
        if self.show_progress:
            frame_pbar = tqdm(
                total=timesteps,
                desc=f"frames:{task_name}/{video_path.stem}",
                leave=False,
                dynamic_ncols=True,
            )

        try:
            for batch_start in range(0, timesteps, self.gen_batch_size):
                batch_end = min(batch_start + self.gen_batch_size, timesteps)
                batch_specs = step_specs[batch_start:batch_end]

                unique_indices = _collect_unique_indices(batch_specs)
                frame_map = self._load_frame_map(vr, unique_indices)

                batch_items: list[dict[str, object]] = []
                for start_idx, end_idx, sampled_indices in batch_specs:
                    images = [frame_map[idx] for idx in sampled_indices]
                    batch_items.append(
                        {
                            "task_name": task_name,
                            "images": images,
                            "start_idx": start_idx,
                            "end_idx": end_idx,
                        }
                    )

                batch_lines = self.generator.generate_lines_batch(batch_items)
                if len(batch_lines) != len(batch_specs):
                    raise RuntimeError(
                        f"Batch output size mismatch: expected={len(batch_specs)}, got={len(batch_lines)}"
                    )
                for (start_idx, _, _), line in zip(batch_specs, batch_lines):
                    lines[start_idx] = line

                if frame_pbar is not None:
                    frame_pbar.update(len(batch_specs))
        finally:
            if frame_pbar is not None:
                frame_pbar.close()

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return True, timesteps

    def _collect_task_units(self) -> list[tuple[str, Path]]:
        """Collect (subset, task_dir) units that have candidate input files."""
        units: list[tuple[str, Path]] = []
        for subset in self.subsets:
            subset_dir = self.target_root / subset
            if not subset_dir.exists():
                logger.warning("Subset directory not found, skip: %s", subset_dir)
                continue

            for task_dir in _iter_task_dirs(subset_dir):
                input_dir = task_dir / self.input_dir_name
                if not input_dir.exists():
                    continue
                input_files = sorted(input_dir.glob("*.mp4"), key=lambda p: _natural_key(p.name))
                if not input_files:
                    continue
                units.append((subset, task_dir))
        return units

    def run_on_task_units(
        self,
        task_units: list[tuple[str, Path]],
        worker_name: str = "single",
    ) -> tuple[int, BackfillStats]:
        stats = BackfillStats()

        logger.info(
            "[%s] Language image backfill started: target_root=%s subsets=%s window_size=%d input_dir=%s output_dir=%s overwrite=%s tasks=%d",
            worker_name,
            self.target_root,
            ",".join(self.subsets),
            self.window_size,
            self.input_dir_name,
            self.output_dir_name,
            self.overwrite,
            len(task_units),
        )
        logger.info(
            "[%s] Runtime options: gen_batch_size=%d video_num_threads=%d",
            worker_name,
            self.gen_batch_size,
            self.video_num_threads,
        )

        task_iter = task_units
        if self.show_progress:
            task_iter = tqdm(task_units, desc=f"tasks:{worker_name}", leave=False, dynamic_ncols=True)

        for subset, task_dir in task_iter:
            stats.tasks_scanned += 1
            input_dir = task_dir / self.input_dir_name
            output_dir = task_dir / self.output_dir_name
            input_files = sorted(input_dir.glob("*.mp4"), key=lambda p: _natural_key(p.name))

            episode_iter = input_files
            if self.show_progress:
                episode_iter = tqdm(
                    input_files,
                    desc=f"episodes:{worker_name}:{subset}/{task_dir.name}",
                    leave=False,
                    dynamic_ncols=True,
                )

            for input_path in episode_iter:
                stats.episodes_total += 1
                out_path = output_dir / f"{input_path.stem}.txt"

                if out_path.exists() and not self.overwrite:
                    stats.episodes_skipped_existing += 1
                    continue

                try:
                    ok, num_lines = self._process_episode(task_name=task_dir.name, video_path=input_path, out_path=out_path)
                    if ok:
                        stats.episodes_generated += 1
                        stats.generated_lines += num_lines
                except Exception as err:  # noqa: BLE001
                    stats.episodes_failed += 1
                    logger.error("[%s] Failed episode: %s error=%s", worker_name, input_path, err)

        logger.info("[%s] Language image backfill finished.", worker_name)
        logger.info("[%s]   tasks_scanned=%d", worker_name, stats.tasks_scanned)
        logger.info("[%s]   episodes_total=%d", worker_name, stats.episodes_total)
        logger.info("[%s]   episodes_generated=%d", worker_name, stats.episodes_generated)
        logger.info("[%s]   episodes_skipped_existing=%d", worker_name, stats.episodes_skipped_existing)
        logger.info("[%s]   episodes_failed=%d", worker_name, stats.episodes_failed)
        logger.info("[%s]   generated_lines=%d", worker_name, stats.generated_lines)

        return (0 if stats.episodes_failed == 0 else 1), stats

    def run(self) -> tuple[int, BackfillStats]:
        task_units = self._collect_task_units()
        return self.run_on_task_units(task_units=task_units, worker_name="single")


def _parse_gpu_devices(gpu_ids: str) -> list[str]:
    devices: list[str] = []
    for tok in gpu_ids.split(","):
        t = tok.strip()
        if not t:
            continue
        if t.startswith("cuda:") or t == "cpu":
            devices.append(t)
        elif t.isdigit():
            devices.append(f"cuda:{t}")
        else:
            raise ValueError(f"Invalid gpu id token: {t}")
    if not devices:
        raise ValueError("No valid GPU ids parsed from --gpu-ids")
    return devices


def _split_task_units(task_units: list[tuple[str, Path]], num_shards: int) -> list[list[tuple[str, Path]]]:
    shards: list[list[tuple[str, Path]]] = [[] for _ in range(num_shards)]
    for idx, unit in enumerate(task_units):
        shards[idx % num_shards].append(unit)
    return shards


def _worker_entry(
    worker_id: int,
    device: str,
    task_units: list[tuple[str, Path]],
    kwargs: dict,
    queue: mp.Queue,
) -> None:
    try:
        runner = RobotWinLanguageImageBackfill(
            target_root=Path(kwargs["target_root"]),
            subsets=kwargs["subsets"],
            vlm_checkpoint_path=kwargs["vlm_checkpoint_path"],
            window_size=kwargs["window_size"],
            input_dir_name=kwargs["input_dir_name"],
            output_dir_name=kwargs["output_dir_name"],
            device=device,
            dtype=kwargs["dtype"],
            max_new_tokens=kwargs["max_new_tokens"],
            temperature=kwargs["temperature"],
            top_p=kwargs["top_p"],
            gen_batch_size=kwargs["gen_batch_size"],
            video_num_threads=kwargs["video_num_threads"],
            show_progress=kwargs["show_progress"],
            overwrite=kwargs["overwrite"],
        )
        code, stats = runner.run_on_task_units(task_units=task_units, worker_name=f"w{worker_id}:{device}")
        queue.put((worker_id, code, stats))
    except Exception as err:  # noqa: BLE001
        logger.exception("Worker %d failed on %s: %s", worker_id, device, err)
        queue.put((worker_id, 1, BackfillStats()))


def run_multi_gpu(args: argparse.Namespace, subsets: list[str]) -> int:
    devices = _parse_gpu_devices(args.gpu_ids)
    logger.info("Using multi-GPU mode, devices=%s", devices)

    # Probe tasks once in parent process.
    probe_runner = RobotWinLanguageImageBackfill(
        target_root=args.target_root,
        subsets=subsets,
        vlm_checkpoint_path=args.vlm_checkpoint_path,
        window_size=args.window_size,
        input_dir_name=args.input_dir_name,
        output_dir_name=args.output_dir_name,
        device=devices[0],
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        gen_batch_size=args.gen_batch_size,
        video_num_threads=args.video_num_threads,
        show_progress=False,
        overwrite=args.overwrite,
    )
    all_units = probe_runner._collect_task_units()
    if not all_units:
        logger.warning("No task units found to process.")
        return 0

    shards = _split_task_units(all_units, len(devices))
    for i, shard in enumerate(shards):
        logger.info("worker %d (%s): %d tasks", i, devices[i], len(shard))

    ctx = mp.get_context("spawn")
    q: mp.Queue = ctx.Queue()
    procs: list[mp.Process] = []
    worker_kwargs = {
        "target_root": str(args.target_root),
        "subsets": subsets,
        "vlm_checkpoint_path": args.vlm_checkpoint_path,
        "window_size": args.window_size,
        "input_dir_name": args.input_dir_name,
        "output_dir_name": args.output_dir_name,
        "dtype": args.dtype,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "gen_batch_size": args.gen_batch_size,
        "video_num_threads": args.video_num_threads,
        "show_progress": not args.no_progress,
        "overwrite": args.overwrite,
    }

    for wid, device in enumerate(devices):
        p = ctx.Process(
            target=_worker_entry,
            args=(wid, device, shards[wid], worker_kwargs, q),
            daemon=False,
        )
        p.start()
        procs.append(p)

    merged = BackfillStats()
    any_fail = False
    for _ in procs:
        wid, code, st = q.get()
        logger.info("worker %d done, code=%d", wid, code)
        merged.merge(st)
        if code != 0:
            any_fail = True

    for p in procs:
        p.join()
        if p.exitcode not in (0, None):
            any_fail = True

    logger.info("Multi-GPU merge summary:")
    logger.info("  tasks_scanned=%d", merged.tasks_scanned)
    logger.info("  episodes_total=%d", merged.episodes_total)
    logger.info("  episodes_generated=%d", merged.episodes_generated)
    logger.info("  episodes_skipped_existing=%d", merged.episodes_skipped_existing)
    logger.info("  episodes_failed=%d", merged.episodes_failed)
    logger.info("  generated_lines=%d", merged.generated_lines)

    if any_fail or merged.episodes_failed > 0:
        return 1
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill language_image text files for RobotWin dataset.",
    )
    parser.add_argument(
        "--target-root",
        type=Path,
        default=Path("./data/robotwin_dataset"),
        help="Root directory of converted RobotWin dataset.",
    )
    parser.add_argument(
        "--subsets",
        type=str,
        default="clean,randomized",
        help="Comma-separated subset names to process.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=16,
        help="Sliding window size for multi-frame scene-change description.",
    )
    parser.add_argument(
        "--input-dir-name",
        type=str,
        default="videos",
        help="Input directory name under each task (default: videos).",
    )
    parser.add_argument(
        "--output-dir-name",
        type=str,
        default="language_image",
        help="Output directory name under each task.",
    )
    parser.add_argument(
        "--vlm-checkpoint-path",
        type=str,
        default="./pretrained_models/Qwen3-VL-2B-Instruct",
        help="Qwen-VL checkpoint path or HF model id.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Inference device, e.g. cuda:0 or cpu.",
    )
    parser.add_argument(
        "--gpu-ids",
        type=str,
        default="",
        help="Comma-separated GPU ids/devices for multi-GPU sharding, e.g. '0,1,2,3' or 'cuda:0,cuda:1'.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
        help="Model dtype for loading.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=96,
        help="Max new tokens per generated line.",
    )
    parser.add_argument(
        "--gen-batch-size",
        type=int,
        default=1,
        help="Micro-batch size for per-episode line generation. 1 keeps legacy behavior.",
    )
    parser.add_argument(
        "--video-num-threads",
        type=int,
        default=2,
        help="Decord VideoReader num_threads value used when decoding video frames.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature. 0 means greedy decoding.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p used when temperature > 0.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing language_image/*.txt files.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    subsets = [s.strip() for s in args.subsets.split(",") if s.strip()]

    try:
        if args.gpu_ids.strip():
            code = run_multi_gpu(args, subsets)
        else:
            runner = RobotWinLanguageImageBackfill(
                target_root=args.target_root,
                subsets=subsets,
                vlm_checkpoint_path=args.vlm_checkpoint_path,
                window_size=args.window_size,
                input_dir_name=args.input_dir_name,
                output_dir_name=args.output_dir_name,
                device=args.device,
                dtype=args.dtype,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                gen_batch_size=args.gen_batch_size,
                video_num_threads=args.video_num_threads,
                show_progress=not args.no_progress,
                overwrite=args.overwrite,
            )
            code, _ = runner.run()
        sys.exit(code)
    except Exception as err:  # noqa: BLE001
        logger.error("Language image backfill failed: %s", err)
        import traceback

        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()

