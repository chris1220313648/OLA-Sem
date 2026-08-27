# OLA-SEM

OLA-SEM is a focused open-source release for latent-action-prompt (LAP)
training and RoboTwin 2.0 inference. It is derived from
[Motus](https://github.com/thu-ml/Motus) and keeps the original `Motus` Python
class names and `policy/Motus` deployment directory so existing checkpoints
remain compatible.

This repository contains source code only. It does **not** include datasets,
pretrained backbones, checkpoints, experiment output, or RoboTwin itself.

## Contents

- `models/`, `train/`, `utils/`, `bak/wan/`: LAP model and training runtime.
- `data/robotwin2/`: RoboTwin loader and conversion utilities (code only).
- `configs/`: RoboTwin LAP, IDM, history-flow, and future-noise examples.
- `inference/robotwin/Motus/`: self-contained RoboTwin policy deployment.

Bridge, DROID, Fractal, LIBERO, and real-world inference are intentionally out
of scope for this release.

## Installation

Python 3.10 and a CUDA-capable PyTorch installation are recommended. Install
PyTorch for your CUDA version first, then install the remaining dependencies:

```bash
conda create -n ola-sem python=3.10 -y
conda activate ola-sem
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
pip install flash-attn --no-build-isolation
pip install -r requirements.txt
```

Download the external model assets into the paths referenced by the training
configs. The default relative layout is:

```text
pretrained_models/
├── Motus/
├── Qwen3-VL-2B-Instruct/
└── Wan2.2-TI2V-5B/
    └── Wan2.2_VAE.pth
```

The upstream Motus README lists the corresponding Hugging Face repositories.
Review their licenses before redistributing any weights.

## RoboTwin data

No data is checked into this repository. The default training path is
`data/robotwin_dataset`. Prepare it with the Python utilities documented in
[`data/robotwin2/robotwin_data_convert/README.md`](data/robotwin2/robotwin_data_convert/README.md),
or point `dataset.dataset_dir` in a config at an existing converted dataset.

The loader expects `clean/` and/or `randomized/` task directories containing
videos, qpos tensors, instruction metadata, and optional cached language
features. The converter README contains the concrete tree.

## LAP training

Edit one of the YAML files in `configs/` to set the dataset and model paths,
then run:

```bash
CONFIG_FILE=configs/robotwin_lap.yaml \
NUM_GPUS=8 \
bash scripts/train_lap.sh
```

Useful overrides are `DEEPSPEED_CONFIG`, `OUTPUT_DIR`, `RUN_NAME`,
`MASTER_ADDR`, `MASTER_PORT`, and `REPORT_TO`. Available examples include:

- `robotwin_lap.yaml`: standard LAP training.
- `robotwin_lap_clean.yaml`: clean split.
- `robotwin_lap_idm.yaml`: IK-style language-action sampling.
- `robotwin_lap_history_flow*.yaml`: executed-qpos history as action source.
- `robotwin_lap_future_noise.yaml`: history initialization with future-video
  noise augmentation.

Set `resume.checkpoint_path` for a full Accelerator/DeepSpeed state, or
`finetune.checkpoint_path` for model-weight initialization. Training writes a
`config.json` beside exported model weights. History-flow inference requires
that metadata and validates `flow_source.mode`, `video_mode`, and
`history_length` against the action chunk size.

## RoboTwin inference

Install RoboTwin 2.0 separately, then deploy the policy directory:

```bash
cp -a inference/robotwin/Motus /path/to/RoboTwin/policy/
cp inference/robotwin/Motus/paths_config.example.yml \
   inference/robotwin/Motus/paths_config.yml
```

Edit `paths_config.yml`. The checkpoint value normally points to an exported
`pytorch_model` directory whose parent contains `config.json`.

Run a single task from this repository:

```bash
bash inference/robotwin/Motus/eval.sh hanging_mug
```

Run every task listed by `tasks_file` across configured or detected GPUs:

```bash
bash inference/robotwin/Motus/auto_eval.sh
```

Both commands accept `CONFIG_FILE=/absolute/path/to/paths_config.yml`. The
single-task entry also accepts `GPU_ID`, `TEST_NUM`, and other variables from
the example config as environment overrides.

## Verification

Lightweight checks do not require model weights, RoboTwin, or a GPU:

```bash
pytest -q
python -m compileall -q models train utils data inference
find scripts inference data -name '*.sh' -print0 | xargs -0 -n1 bash -n
```

Full training and simulator evaluation require external models, converted data,
RoboTwin, and suitable GPUs; they are not exercised by the lightweight suite.

## License and attribution

This release is provided under the Apache License 2.0 in `LICENSE`. See
`NOTICE` for upstream attribution. When using the retained Motus implementation,
please also follow the citation guidance of the upstream project.
