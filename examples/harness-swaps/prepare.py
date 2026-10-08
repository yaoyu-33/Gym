# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Select unchanged demo tasks from already prepared benchmark data."""

import argparse
import hashlib
import json
from pathlib import Path

from run import ROOT, select_tasks, task_id


SWE_TASK = (
    "instance_ansible__ansible-395e5e20fab9cad517243372fa3c3c5d9e09ab2a-v7eee2454f617569fd6889f2211f75bc02a35f9f8"
)


def prepare(source: Path, destination: Path, *, ids: list[str]) -> None:
    """Write an explicit selection and provenance, refusing to overwrite prior inputs."""
    rows = select_tasks(source, limit=len(ids), task_ids=ids)
    # Saved materialized inputs can contain old execution identity. No task fields are dropped.
    rows = [
        {key: value for key, value in row.items() if not key.startswith("_ng_") and key != "agent_ref"} for row in rows
    ]
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    manifest = {
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "selected_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        "tasks": [task_id(row) for row in rows],
        "removed_execution_keys": ["_ng_*", "agent_ref"],
    }
    destination.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Prepared {len(rows)} task(s): {destination}")


def main() -> None:
    """Select the rehearsal tasks or user-specified task IDs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--swe-source", type=Path, default=ROOT / "benchmarks/swebench/data/swebench_pro_benchmark.jsonl"
    )
    parser.add_argument("--tb-source", type=Path, default=ROOT / "benchmarks/terminal_bench_2_1/data/benchmark.jsonl")
    parser.add_argument("--swe-task", action="append")
    parser.add_argument("--tb-task", action="append")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "data")
    args = parser.parse_args()
    prepare(args.swe_source, args.output / "swe-pro.jsonl", ids=args.swe_task or [SWE_TASK])
    prepare(args.tb_source, args.output / "tb21.jsonl", ids=args.tb_task or ["terminal-bench/regex-log"])


if __name__ == "__main__":
    main()
