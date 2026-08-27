# RoboTwin data conversion

These utilities convert RoboTwin 2.0 episodes into the format consumed by the
OLA-SEM loader. They are source code only; no downloaded or converted data is
included.

## Download and convert

```bash
python download_robotwin_dataset.py --output_dir ./data/robotwin_raw_dataset

# Edit source_root, target_root, and wan_repo_path first.
python robotwin_converter.py --config config.yml
```

Use `python <script> --help` for the optional epos, language-action, language
image, and T5 cache generators.

## Input layout

```text
robotwin_raw_dataset/
└── <task>/
    ├── aloha-agilex_clean_50/
    │   ├── data/episode*.hdf5
    │   └── instructions/episode*.json
    └── aloha-agilex_randomized_50/
        ├── data/episode*.hdf5
        └── instructions/episode*.json
```

## Output layout

```text
robotwin_dataset/
├── clean/<task>/
│   ├── videos/0.mp4
│   ├── qpos/0.pt
│   ├── metas/0.txt
│   └── umt5_wan/0.pt
└── randomized/<task>/
    ├── videos/
    ├── qpos/
    ├── metas/
    └── umt5_wan/
```

Generated `.hdf5`, `.mp4`, and `.pt` files are excluded by the repository
`.gitignore`.
