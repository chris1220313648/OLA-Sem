import importlib.util
from pathlib import Path

import pytest
import torch


MODULE_PATH = (
    Path(__file__).parents[1]
    / "inference"
    / "robotwin"
    / "Motus"
    / "utils"
    / "inference_schedule.py"
)
SPEC = importlib.util.spec_from_file_location("motus_inference_schedule", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
build_joint_denoising_timesteps = MODULE.build_joint_denoising_timesteps


@pytest.mark.parametrize(
    ("fraction", "expected_video_end"),
    [(0.0, 1.0), (0.5, 0.5), (1.0, 0.0)],
)
def test_build_joint_denoising_timesteps(fraction, expected_video_end):
    video, action = build_joint_denoising_timesteps(
        4,
        fraction,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert video.shape == action.shape == (5,)
    assert video[0].item() == pytest.approx(1.0)
    assert video[-1].item() == pytest.approx(expected_video_end)
    assert torch.equal(action, torch.linspace(1.0, 0.0, 5))


@pytest.mark.parametrize("fraction", [-0.01, 1.01])
def test_build_joint_denoising_timesteps_rejects_invalid_fraction(fraction):
    with pytest.raises(ValueError, match="must be in \\[0, 1\\]"):
        build_joint_denoising_timesteps(
            4,
            fraction,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )


def test_build_joint_denoising_timesteps_rejects_non_positive_steps():
    with pytest.raises(ValueError, match="must be positive"):
        build_joint_denoising_timesteps(
            0,
            0.5,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
