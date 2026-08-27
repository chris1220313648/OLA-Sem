from __future__ import annotations

import os
import subprocess
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_PATHS = (
    "README.md",
    "LICENSE",
    "requirements.txt",
    "models/motus.py",
    "train/train.py",
    "data/dataset.py",
    "scripts/train_lap.sh",
    "configs/robotwin_lap.yaml",
    "inference/robotwin/Motus/deploy_policy.py",
    "inference/robotwin/Motus/paths_config.example.yml",
    "inference/robotwin/Motus/eval.sh",
    "inference/robotwin/Motus/auto_eval.sh",
)
PROHIBITED_PATH_PARTS = {
    "checkpoints",
    "pretrained_models",
    "outputs",
    "wandb",
    "logs",
}
PROHIBITED_SUFFIXES = {
    ".hdf5",
    ".mp4",
    ".npy",
    ".npz",
    ".pt",
    ".pth",
    ".safetensors",
}


def test_release_contains_runnable_training_and_inference_entrypoints():
    missing = [path for path in REQUIRED_PATHS if not (REPO_ROOT / path).is_file()]
    assert not missing, f"missing release files: {missing}"


def test_release_yaml_is_parseable_and_portable():
    yaml_paths = sorted(REPO_ROOT.rglob("*.yaml")) + sorted(REPO_ROOT.rglob("*.yml"))
    assert yaml_paths, "release must contain YAML configuration"

    for path in yaml_paths:
        text = path.read_text(encoding="utf-8")
        assert "/data/user/" not in text, path
        assert "/share/" not in text, path
        assert yaml.safe_load(text) is not None, path


def test_release_shell_entrypoints_are_syntactically_valid():
    shell_paths = sorted(REPO_ROOT.rglob("*.sh"))
    assert shell_paths, "release must contain shell entrypoints"

    for path in shell_paths:
        assert os.access(path, os.X_OK), f"shell entrypoint is not executable: {path}"
        result = subprocess.run(
            ["bash", "-n", str(path)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, f"{path}: {result.stderr}"


def test_release_excludes_runtime_data_and_model_artifacts():
    violations = []
    for path in REPO_ROOT.rglob("*"):
        if path == REPO_ROOT / ".git" or ".git" in path.parts:
            continue
        relative = path.relative_to(REPO_ROOT)
        if any(part in PROHIBITED_PATH_PARTS for part in relative.parts):
            violations.append(str(relative))
        elif path.is_file() and path.suffix.lower() in PROHIBITED_SUFFIXES:
            violations.append(str(relative))

    assert not violations, f"prohibited release artifacts: {violations}"


def test_release_text_excludes_private_cluster_paths():
    prohibited_fragments = (
        "/data" + "/user/",
        "/sha" + "re/",
        "#SBATCH",
        "lihaoang" + "_rent",
        "acd" + "_u",
    )
    violations = []
    for path in REPO_ROOT.rglob("*"):
        relative = path.relative_to(REPO_ROOT)
        if (
            not path.is_file()
            or ".git" in path.parts
            or "__pycache__" in path.parts
            or relative.parts[:1] == ("tests",)
        ):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        matches = [fragment for fragment in prohibited_fragments if fragment in text]
        if matches:
            violations.append(f"{relative}: {matches}")

    assert not violations, "private path references found: " + "; ".join(violations)
