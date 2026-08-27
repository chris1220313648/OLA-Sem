import pytest
import torch

from utils.video_noise_augmentation import apply_future_frame_noise_augmentation


def test_future_frame_noise_augmentation_preserves_condition_frame():
    latents = torch.arange(2 * 3 * 4 * 1 * 1, dtype=torch.float32).view(2, 3, 4, 1, 1)
    noise = torch.full((2, 3, 3, 1, 1), 10.0)
    scales = torch.full((2, 1, 1, 1, 1), 0.75)

    augmented, info = apply_future_frame_noise_augmentation(
        latents,
        enabled=True,
        probability=1.0,
        min_scale=0.5,
        max_scale=1.0,
        future_start_index=1,
        noise=noise,
        scales=scales,
    )

    expected_future = (1.0 - scales) * noise + scales * latents[:, :, 1:]

    torch.testing.assert_close(augmented[:, :, :1], latents[:, :, :1])
    torch.testing.assert_close(augmented[:, :, 1:], expected_future)
    assert info["applied"].all()
    torch.testing.assert_close(info["scales"], scales)


def test_future_frame_noise_augmentation_can_be_disabled_by_probability():
    latents = torch.randn(2, 3, 4, 2, 2)

    augmented, info = apply_future_frame_noise_augmentation(
        latents,
        enabled=True,
        probability=0.0,
    )

    torch.testing.assert_close(augmented, latents)
    assert not info["applied"].any()


def test_future_frame_noise_augmentation_can_be_disabled_by_flag():
    latents = torch.randn(2, 3, 4, 2, 2)

    augmented, info = apply_future_frame_noise_augmentation(
        latents,
        enabled=False,
        probability=1.0,
    )

    torch.testing.assert_close(augmented, latents)
    assert not info["applied"].any()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"probability": -0.1}, "probability"),
        ({"probability": 1.1}, "probability"),
        ({"min_scale": 0.8, "max_scale": 0.7}, "scale range"),
        ({"future_start_index": 0}, "future_start_index"),
    ],
)
def test_future_frame_noise_augmentation_validates_parameters(kwargs, message):
    latents = torch.randn(2, 3, 4, 2, 2)

    with pytest.raises(ValueError, match=message):
        apply_future_frame_noise_augmentation(
            latents,
            enabled=True,
            **kwargs,
        )
