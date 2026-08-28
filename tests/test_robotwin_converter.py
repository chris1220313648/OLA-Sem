from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
CONVERTER_PATH = (
    REPO_ROOT / "data/robotwin2/robotwin_data_convert/robotwin_converter.py"
)


@pytest.fixture(scope="module")
def converter_class():
    """Load the scanner without requiring the optional GPU/data dependencies."""
    torch_stub = types.ModuleType("torch")
    torch_stub.Tensor = type("Tensor", (), {})
    torch_stub.bfloat16 = object()

    dependency_stubs = {
        "torch": torch_stub,
        "h5py": types.ModuleType("h5py"),
        "cv2": types.ModuleType("cv2"),
    }
    spec = importlib.util.spec_from_file_location(
        "_ola_sem_robotwin_converter_test", CONVERTER_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, dependency_stubs):
        spec.loader.exec_module(module)
    return module.RobotWinConverter


def _converter(converter_class, mapping=None):
    converter = converter_class.__new__(converter_class)
    converter.config = {"demo_type_mapping": mapping or {}}
    return converter


def _episode(root: Path, task: str, demo: str, name: str) -> Path:
    path = root / task / demo / "data" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return path


def test_scan_task_first_source_and_sort_episodes(tmp_path, converter_class):
    episode_2 = _episode(
        tmp_path, "beat_block_hammer", "aloha-agilex_clean_50", "episode2.hdf5"
    )
    episode_1 = _episode(
        tmp_path, "beat_block_hammer", "aloha-agilex_clean_50", "episode1.hdf5"
    )
    randomized = _episode(
        tmp_path,
        "beat_block_hammer",
        "aloha-agilex_randomized_500",
        "episode0.hdf5",
    )
    _episode(tmp_path, "beat_block_hammer", "unrecognized", "ignored.hdf5")

    structure = _converter(converter_class).scan_dataset(str(tmp_path))

    assert structure == {
        "clean": {"beat_block_hammer": [episode_1, episode_2]},
        "randomized": {"beat_block_hammer": [randomized]},
    }


def test_scan_honors_mapping_and_downloader_wrapper(tmp_path, converter_class):
    wrapped_root = tmp_path / "dataset"
    episode = _episode(wrapped_root, "click_bell", "custom_demo", "episode0.hdf5")

    converter = _converter(converter_class, {"custom_demo": "clean"})
    structure = converter.scan_dataset(str(tmp_path))

    assert structure == {"clean": {"click_bell": [episode]}}


def test_conversion_still_runs_t5_cache_stage(tmp_path, converter_class):
    converter = _converter(converter_class)
    converter.config.update(
        {
            "source_root": str(tmp_path / "raw"),
            "target_root": str(tmp_path / "converted"),
        }
    )
    episode = tmp_path / "raw/task/demo/data/episode0.hdf5"
    converter.scan_dataset = lambda _root: {"clean": {"task": [episode]}}

    processed = []
    converter.process_episode = lambda *args: processed.append(args) or True
    t5_calls = []
    converter.process_t5_embeddings_parallel = lambda: t5_calls.append(True)

    converter.convert_dataset()

    assert len(processed) == 1
    assert processed[0][2] == tmp_path / "converted/clean/task"
    assert t5_calls == [True]
