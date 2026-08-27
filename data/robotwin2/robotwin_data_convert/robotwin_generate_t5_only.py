#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate umt5_wan embeddings only for an already-converted RobotWin dataset.

This script scans:
  target_root/{clean,randomized}/{task}/metas/*.txt
and writes:
  target_root/{clean,randomized}/{task}/umt5_wan/*.pt

It does NOT run any video/qpos/meta conversion.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

try:
    # Running as script from the same directory
    from robotwin_converter import T5EmbeddingProcessor, process_t5_batch
except ModuleNotFoundError:
    # Running from project root / package import style
    from data.robotwin2.robotwin_data_convert.robotwin_converter import (  # type: ignore
        T5EmbeddingProcessor,
        process_t5_batch,
    )


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class RobotWinT5Backfill:
    """Backfill T5 embeddings under target_root only."""

    def __init__(self, config_path: str, overwrite: bool = False):
        self.config_path = config_path
        self.overwrite = overwrite
        self.config = self._load_config(config_path)
        self._validate_config()

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
            logger.info(f"Loaded configuration from: {config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load config file {config_path}: {e}")
            raise

    def _validate_config(self) -> None:
        if "target_root" not in self.config:
            raise ValueError("Required config key missing: target_root")

        target_root = Path(self.config["target_root"])
        if not target_root.exists():
            raise FileNotFoundError(f"target_root not found: {target_root}")

        wan_repo_path = self.config.get("wan_repo_path", "")
        if not wan_repo_path:
            raise ValueError("Required config key missing or empty: wan_repo_path")

        cuda_devices = self.config.get("cuda_devices", ["0"])
        if isinstance(cuda_devices, str):
            cuda_devices = [x.strip() for x in cuda_devices.split(",") if x.strip()]
        if not cuda_devices:
            raise ValueError("cuda_devices is empty in config")

    def collect_meta_files_for_t5(self) -> Tuple[List[Tuple[str, str]], int]:
        """
        Collect meta->t5 pairs to process.

        Returns:
            pending_pairs: list[(meta_file, t5_output_file)]
            skipped_existing: count of skipped files due to existing output
        """
        target_root = Path(self.config["target_root"])
        pending_pairs: List[Tuple[str, str]] = []
        skipped_existing = 0

        for subset in ("clean", "randomized"):
            subset_dir = target_root / subset
            if not subset_dir.exists():
                logger.warning(f"Subset directory not found, skip: {subset_dir}")
                continue

            for task_dir in subset_dir.iterdir():
                if not task_dir.is_dir():
                    continue

                metas_dir = task_dir / "metas"
                if not metas_dir.exists():
                    logger.warning(f"metas directory not found, skip: {metas_dir}")
                    continue

                umt5_dir = task_dir / "umt5_wan"
                umt5_dir.mkdir(exist_ok=True)

                for meta_file in metas_dir.glob("*.txt"):
                    t5_file = umt5_dir / f"{meta_file.stem}.pt"
                    if t5_file.exists() and not self.overwrite:
                        skipped_existing += 1
                        continue
                    pending_pairs.append((str(meta_file), str(t5_file)))

        return pending_pairs, skipped_existing

    def run(self) -> int:
        # Intentionally ignore enable_t5_embeddings; this script is T5-only backfill.
        wan_repo_path = self.config.get("wan_repo_path", "")
        t5_max_length = int(self.config.get("t5_max_length", 512))
        cuda_devices = self.config.get("cuda_devices", ["0"])
        if isinstance(cuda_devices, str):
            cuda_devices = [x.strip() for x in cuda_devices.split(",") if x.strip()]

        pending_pairs, skipped_existing = self.collect_meta_files_for_t5()
        total_pending = len(pending_pairs)
        logger.info(
            f"T5 backfill scan completed: pending={total_pending}, "
            f"skipped_existing={skipped_existing}, overwrite={self.overwrite}"
        )

        if total_pending == 0:
            logger.info("No pending T5 files to generate.")
            return 0

        num_devices = len(cuda_devices)
        logger.info(f"Using {num_devices} GPUs: {cuda_devices}")

        chunks = [pending_pairs[i::num_devices] for i in range(num_devices)]
        processors_and_chunks = []
        for i, device_id in enumerate(cuda_devices):
            processor = T5EmbeddingProcessor(
                wan_repo_path=wan_repo_path,
                t5_max_length=t5_max_length,
                device=f"cuda:{device_id}",
            )
            processors_and_chunks.append((processor, chunks[i]))

        all_results: List[Tuple[str, bool]] = []
        with ProcessPoolExecutor(max_workers=num_devices) as executor:
            futures = [executor.submit(process_t5_batch, args) for args in processors_and_chunks]
            for future in tqdm(futures, desc="Processing T5 embeddings"):
                results = future.result()
                all_results.extend(results)

        successful = sum(1 for _, ok in all_results if ok)
        failed = total_pending - successful
        logger.info(
            f"T5 backfill completed: success={successful}, failed={failed}, "
            f"skipped_existing={skipped_existing}"
        )

        if failed > 0:
            failed_files = [path for path, ok in all_results if not ok]
            logger.warning(f"Failed files (first 20): {failed_files[:20]}")
            return 1
        return 0


def parse_args() -> argparse.Namespace:
    default_cfg = str(Path(__file__).with_name("config.yml"))
    parser = argparse.ArgumentParser(
        description="Backfill only umt5_wan embeddings for an already converted dataset."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=default_cfg,
        help="Path to configuration YAML file (default: script-dir/config.yml)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate all embeddings even if umt5_wan/*.pt already exists.",
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

    try:
        runner = RobotWinT5Backfill(config_path=args.config, overwrite=args.overwrite)
        code = runner.run()
        sys.exit(code)
    except Exception as e:
        logger.error(f"T5 backfill failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
