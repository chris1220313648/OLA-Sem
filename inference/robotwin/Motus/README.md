# OLA-SEM Policy Evaluation on RoboTwin

This guide explains how to deploy and evaluate OLA-SEM on RoboTwin 2.0. The
policy directory keeps the internal name `Motus` for compatibility with
RoboTwin and existing checkpoints.

## Environment

Install RoboTwin 2.0 by following its official installation guide. A separate
RoboTwin environment is recommended because simulator and training dependencies
may differ.

After activating the RoboTwin environment, install the policy dependencies:

```bash
cd /path/to/OLA-Sem/inference/robotwin/Motus
pip install -r requirements.txt
```

## Deployment

Copy the complete policy directory into RoboTwin:

```bash
cp -a /path/to/OLA-Sem/inference/robotwin/Motus \
  /path/to/RoboTwin/policy/
cd /path/to/RoboTwin/policy/Motus
cp paths_config.example.yml paths_config.yml
```

The deployed layout should include:

```text
RoboTwin/
├── policy/Motus/
│   ├── deploy_policy.py
│   ├── deploy_policy.yml
│   ├── eval.sh
│   ├── auto_eval.sh
│   ├── paths_config.yml
│   ├── tasks_all.txt
│   ├── models/
│   ├── utils/
│   └── bak/wan/
└── script/eval_policy.py
```

## Configuration

Edit `paths_config.yml` before evaluation:

```yaml
robotwin_root: "/path/to/RoboTwin"
conda_env: "/path/to/conda/envs/RoboTwin"
checkpoint_path: "/path/to/checkpoint_step_N/pytorch_model"
wan_path: "/path/to/Wan2.2-TI2V-5B"
vlm_path: "/path/to/Qwen3-VL-2B-Instruct"

gpu_ids: []
task_config: "demo_randomized"
seed: 42
tasks_file: "tasks_all.txt"
instruction_type: "unseen"
test_num: 100

inference_mode: "history_flow"
num_inference_timesteps: 4
history_action_noise_std: 0.02
future_video_denoise_fraction: 1.0

save_images: false
image_save_interval: 0
task_timeout_seconds: 3600
```

Required paths:

- `robotwin_root`: the RoboTwin repository root.
- `checkpoint_path`: the exported `pytorch_model/` directory containing
  `mp_rank_00_model_states.pt`.
- `wan_path`: Wan2.2 directory containing the VAE, T5 weights, and tokenizer.
- `vlm_path`: Qwen3-VL directory used to load its configuration and processor.
- `conda_env`: RoboTwin environment path or name. Leave it empty if the correct
  environment is already active.

`gpu_ids: []` enables `nvidia-smi` auto-detection. Set an explicit list such as
`[0, 1, 2, 3]` when needed. Batch evaluation assigns one task to each GPU at a
time.

## Inference modes

### Legacy

```yaml
inference_mode: "legacy"
```

Legacy mode uses Gaussian flow sources and supports checkpoints created before
history-flow training was introduced.

### History flow

```yaml
inference_mode: "history_flow"
```

History-flow mode conditions the next prediction on qpos values actually
executed by the simulator. The checkpoint layout must be:

```text
checkpoint_step_N/
├── config.json
└── pytorch_model/
    └── mp_rank_00_model_states.pt
```

The adjacent `config.json` must contain `flow_source.mode: "history"` and a
`history_length` equal to the action chunk size. Its `video_mode` may be
`gaussian` or `history`. `history_action_noise_std` overrides the checkpoint
noise value and must be non-negative.

`num_inference_timesteps` must be positive.
`future_video_denoise_fraction` must be between `0.0` and `1.0`; `1.0` denoises
the future-video branch throughout the full inference schedule.

## Running evaluation

Evaluate one task:

```bash
cd /path/to/RoboTwin/policy/Motus
bash eval.sh hanging_mug
```

Evaluate all tasks listed in `tasks_all.txt` across the configured GPUs:

```bash
bash auto_eval.sh
```

To use another configuration file:

```bash
CONFIG_FILE=/absolute/path/to/paths_config.yml bash eval.sh hanging_mug
CONFIG_FILE=/absolute/path/to/paths_config.yml bash auto_eval.sh
```

Common values can also be overridden without editing YAML:

```bash
GPU_ID=1 TEST_NUM=10 NUM_INFERENCE_TIMESTEPS=4 \
SAVE_IMAGES=true IMAGE_SAVE_INTERVAL=10 \
bash eval.sh hanging_mug
```

Other supported overrides include `ROBOTWIN_ROOT`, `CONDA_ENV`,
`CHECKPOINT_PATH`, `WAN_PATH`, `VLM_PATH`, `TASK_CONFIG`, `SEED`,
`INSTRUCTION_TYPE`, `INFERENCE_MODE`, `HISTORY_ACTION_NOISE_STD`,
`FUTURE_VIDEO_DENOISE_FRACTION`, `TASK_TIMEOUT_SECONDS`, `TASKS_FILE`, and
`GPU_IDS`.

## Logs and troubleshooting

Each run writes logs to `logs_YYYYMMDD_HHMMSS/` under the policy directory
unless `LOG_DIR` is set. Diagnostic images are disabled by default; when
enabled, they are stored under `<log_dir>/images/<task>/`.

Common failures:

- `configuration not found`: copy `paths_config.example.yml` to
  `paths_config.yml`, or set `CONFIG_FILE`.
- `deploy this directory`: copy the policy to `<robotwin_root>/policy/Motus`.
- `Checkpoint file not found`: ensure `checkpoint_path` points to the
  `pytorch_model/` directory rather than its parent.
- `history_flow requires checkpoint metadata`: place the compatible
  `config.json` beside `pytorch_model/`, as shown above.
- Conda activation failure: activate the RoboTwin environment manually and set
  `conda_env: ""`.

The scripts do not load site-specific CUDA modules. Load any modules required
by your cluster before running evaluation.

This deployment is adapted from the
[Motus RoboTwin evaluation workflow](https://github.com/thu-ml/Motus/blob/main/inference/robotwin/Motus/README.md).
