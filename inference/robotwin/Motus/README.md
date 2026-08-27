# OLA-SEM policy for RoboTwin

This directory is intentionally self-contained and retains the `Motus` policy
name for RoboTwin and checkpoint compatibility.

1. Install RoboTwin 2.0 and its environment.
2. Copy this directory to `<robotwin_root>/policy/Motus`.
3. Copy `paths_config.example.yml` to `paths_config.yml` in the source or
   deployed policy directory and edit every required path.
4. Run `bash eval.sh <task_name>` or `bash auto_eval.sh`.

`history_flow` requires a checkpoint whose parent directory contains
`config.json` with `flow_source.mode: history` and a `history_length` matching
the configured action chunk size.

The scripts do not load site-specific CUDA modules. Activate them externally if
your cluster requires modules. `conda_env` may be left empty when the correct
environment is already active.
