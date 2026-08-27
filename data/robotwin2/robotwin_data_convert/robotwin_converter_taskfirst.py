#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RobotWin Data Converter (Task-First Source Layout)

Input layout:
source_root/
├── {task_name}/
│   ├── aloha-agilex_clean_50/
│   │   ├── data/episode*.hdf5
│   │   └── instructions/episode*.json
│   └── aloha-agilex_randomized_500/
│       ├── data/episode*.hdf5
│       └── instructions/episode*.json

Output layout:
target_root/
├── clean/{task_name}/...
└── randomized/{task_name}/...

All per-episode processing logic is inherited from robotwin_converter.py.
Only dataset scanning is customized for the task-first source structure.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import sys
from pathlib import Path
from typing import Dict, List

# Keep spawn behavior aligned with the original converter.
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

from robotwin_converter import RobotWinConverter, logger, parse_args


class RobotWinTaskFirstConverter(RobotWinConverter):
    """Converter for source_root/{task}/{demo_type}/... layout."""

    def _map_demo_to_subset(self, demo_name: str) -> str | None:
        """Map demo folder name to subset name using config and fallback rules."""
        demo_type_mapping = self.config.get("demo_type_mapping", {}) or {}
        mapped = demo_type_mapping.get(demo_name)
        if mapped:
            return str(mapped)

        lower = demo_name.lower()
        if "clean" in lower:
            return "clean"
        if "randomized" in lower:
            return "randomized"
        return None

    def scan_dataset(self, source_root: str) -> Dict[str, Dict[str, List[Path]]]:
        """
        Scan task-first source layout and return:
        {subset: {task_name: [episode_paths]}}
        """
        dataset_structure: Dict[str, Dict[str, List[Path]]] = {}
        source_path = Path(source_root)

        if not source_path.exists():
            logger.error(f"Source root not found: {source_path}")
            return dataset_structure

        # source_root/{task_name}/{demo_type}/...
        for task_path in source_path.iterdir():
            if not task_path.is_dir():
                continue
            task_name = task_path.name

            for demo_path in task_path.iterdir():
                if not demo_path.is_dir():
                    continue

                subset_name = self._map_demo_to_subset(demo_path.name)
                if subset_name is None:
                    logger.warning(f"Skipping unrecognized demo folder: {demo_path}")
                    continue

                # Find episode files under demo root or demo/data
                hdf5_files = list(demo_path.glob("*.hdf5"))
                if not hdf5_files:
                    data_dir = demo_path / "data"
                    if data_dir.exists():
                        hdf5_files = list(data_dir.glob("*.hdf5"))

                if not hdf5_files:
                    logger.warning(f"No .hdf5 files found in {demo_path}")
                    continue

                dataset_structure.setdefault(subset_name, {}).setdefault(task_name, [])
                dataset_structure[subset_name][task_name].extend(sorted(hdf5_files))

        # Keep deterministic order and emit summary.
        for subset_name, tasks in dataset_structure.items():
            for task_name, files in tasks.items():
                tasks[task_name] = sorted(files)
                logger.info(f"Task {task_name} ({subset_name}): {len(tasks[task_name])} episodes")

        return dataset_structure


def main():
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        converter = RobotWinTaskFirstConverter(args.config)
        converter.convert_dataset()
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
