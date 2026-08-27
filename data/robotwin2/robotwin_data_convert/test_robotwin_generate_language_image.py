#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image


def load_target_module():
    # Provide a lightweight stub so module import does not depend on runtime qwen utils location.
    if "qwen_vl_utils" not in sys.modules:
        stub = types.ModuleType("qwen_vl_utils")
        stub.process_vision_info = lambda messages: (None, None)
        sys.modules["qwen_vl_utils"] = stub

    module_path = Path(__file__).parent / "robotwin_generate_language_image.py"
    spec = importlib.util.spec_from_file_location("rw_lang_image", module_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


class TestRobotWinLanguageImageUtils(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = load_target_module()

    def test_parse_gpu_devices(self):
        self.assertEqual(self.mod._parse_gpu_devices("0,1,2"), ["cuda:0", "cuda:1", "cuda:2"])
        self.assertEqual(self.mod._parse_gpu_devices("cuda:0,cuda:3"), ["cuda:0", "cuda:3"])
        self.assertEqual(self.mod._parse_gpu_devices("cpu"), ["cpu"])

    def test_parse_gpu_devices_invalid(self):
        with self.assertRaises(ValueError):
            self.mod._parse_gpu_devices("")
        with self.assertRaises(ValueError):
            self.mod._parse_gpu_devices("gpu0")

    def test_split_task_units_round_robin(self):
        units = [("clean", Path(f"/tmp/task{i}")) for i in range(7)]
        shards = self.mod._split_task_units(units, 3)
        self.assertEqual(len(shards), 3)
        self.assertEqual([len(s) for s in shards], [3, 2, 2])
        # round-robin assignment checks
        self.assertEqual(shards[0][0][1].name, "task0")
        self.assertEqual(shards[1][0][1].name, "task1")
        self.assertEqual(shards[2][0][1].name, "task2")
        self.assertEqual(shards[0][1][1].name, "task3")

    def test_sample_three_indices(self):
        self.assertEqual(self.mod._sample_three_indices(0, 1), [0])
        self.assertEqual(self.mod._sample_three_indices(5, 6), [5])
        self.assertEqual(self.mod._sample_three_indices(0, 3), [0, 1, 2])
        self.assertEqual(self.mod._sample_three_indices(10, 20), [10, 14, 19])

    def test_collect_task_units(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            # clean/task_a/videos/0.mp4
            (root / "clean" / "task_a" / "videos").mkdir(parents=True, exist_ok=True)
            (root / "clean" / "task_a" / "videos" / "0.mp4").touch()
            # clean/task_b/videos exists but empty => should be skipped
            (root / "clean" / "task_b" / "videos").mkdir(parents=True, exist_ok=True)
            # randomized/task_c has no videos dir => skipped
            (root / "randomized" / "task_c").mkdir(parents=True, exist_ok=True)

            # Construct instance without invoking __init__ (which loads model).
            runner = self.mod.RobotWinLanguageImageBackfill.__new__(self.mod.RobotWinLanguageImageBackfill)
            runner.target_root = root
            runner.subsets = ["clean", "randomized"]
            runner.input_dir_name = "videos"

            units = runner._collect_task_units()
            self.assertEqual(len(units), 1)
            self.assertEqual(units[0][0], "clean")
            self.assertEqual(units[0][1].name, "task_a")

    def test_build_timestep_specs_and_unique_indices(self):
        specs = self.mod._build_timestep_specs(timesteps=3, window_size=16)
        self.assertEqual(specs[0], (0, 3, [0, 1, 2]))
        self.assertEqual(specs[1], (1, 3, [1, 2]))
        self.assertEqual(specs[2], (2, 3, [2]))
        unique = self.mod._collect_unique_indices(specs[:2])
        self.assertEqual(unique, [0, 1, 2])


class _FakeModel:
    def generate(self, **kwargs):
        batch = kwargs["input_ids"].shape[0]
        return [torch.tensor([11, 12, i + 100], dtype=torch.long) for i in range(batch)]


class _FakeProcessorInputs(dict):
    def to(self, _device):
        return self

    @property
    def input_ids(self):
        return self["input_ids"]


class _FakeProcessor:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        _ = tokenize
        _ = add_generation_prompt
        text_items = [x["text"] for x in messages[1]["content"] if x["type"] == "text"]
        return text_items[0]

    def __call__(self, text, images, videos, padding, return_tensors):
        _ = images
        _ = videos
        _ = padding
        _ = return_tensors
        batch = len(text)
        input_ids = torch.ones((batch, 2), dtype=torch.long)
        return _FakeProcessorInputs({"input_ids": input_ids})

    def batch_decode(self, generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False):
        _ = skip_special_tokens
        _ = clean_up_tokenization_spaces
        decoded = []
        for tid in generated_ids_trimmed:
            decoded.append(f"line_{int(tid[-1].item())}")
        return decoded


class _FakeDecordBatch:
    def __init__(self, arr):
        self.arr = arr

    def asnumpy(self):
        return self.arr


class _FakeVideoReader:
    def __init__(self, _path, ctx=None, num_threads=2):
        _ = ctx
        _ = num_threads
        self.frames = [np.full((4, 4, 3), i, dtype=np.uint8) for i in range(5)]

    def __len__(self):
        return len(self.frames)

    def get_batch(self, indices):
        arr = np.stack([self.frames[i] for i in indices], axis=0)
        return _FakeDecordBatch(arr)


class _FakeGenerator:
    def generate_lines_batch(self, batch_items):
        lines = []
        for item in batch_items:
            start_idx = int(item["start_idx"])
            lines.append(f"t{start_idx}")
        return lines


class TestRobotWinLanguageImageBatch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = load_target_module()

    def test_generate_lines_batch_order(self):
        gen = self.mod.QwenVLSceneChangeGenerator.__new__(self.mod.QwenVLSceneChangeGenerator)
        gen.device = "cpu"
        gen.max_new_tokens = 8
        gen.temperature = 0.0
        gen.top_p = 0.9
        gen.processor = _FakeProcessor()
        gen.model = _FakeModel()

        img0 = Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8), mode="RGB")
        img1 = Image.fromarray(np.ones((4, 4, 3), dtype=np.uint8), mode="RGB")
        batch_items = [
            {"task_name": "task_a", "images": [img0], "start_idx": 0, "end_idx": 1},
            {"task_name": "task_a", "images": [img1], "start_idx": 1, "end_idx": 2},
        ]
        lines = gen.generate_lines_batch(batch_items)
        self.assertEqual(lines, ["line_100", "line_101"])

    def test_generate_line_equivalent_to_singleton_batch(self):
        gen = self.mod.QwenVLSceneChangeGenerator.__new__(self.mod.QwenVLSceneChangeGenerator)
        gen.device = "cpu"
        gen.max_new_tokens = 8
        gen.temperature = 0.0
        gen.top_p = 0.9
        gen.processor = _FakeProcessor()
        gen.model = _FakeModel()

        img = Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8), mode="RGB")
        single = gen.generate_line(task_name="task", images=[img], start_idx=3, end_idx=4)
        batch = gen.generate_lines_batch(
            [{"task_name": "task", "images": [img], "start_idx": 3, "end_idx": 4}]
        )[0]
        self.assertEqual(single, batch)

    def test_process_episode_batch_size_equivalence(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            in_path = base / "0.mp4"
            in_path.touch()
            out1 = base / "out_b1.txt"
            out3 = base / "out_b3.txt"

            runner = self.mod.RobotWinLanguageImageBackfill.__new__(self.mod.RobotWinLanguageImageBackfill)
            runner.window_size = 3
            runner.video_num_threads = 2
            runner.show_progress = False
            runner.generator = _FakeGenerator()

            with patch.object(self.mod, "VideoReader", _FakeVideoReader):
                runner.gen_batch_size = 1
                ok1, n1 = runner._process_episode("task", in_path, out1)
                runner.gen_batch_size = 3
                ok3, n3 = runner._process_episode("task", in_path, out3)

            self.assertTrue(ok1 and ok3)
            self.assertEqual(n1, 5)
            self.assertEqual(n3, 5)
            self.assertEqual(out1.read_text(encoding="utf-8"), out3.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
