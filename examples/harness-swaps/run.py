# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Small demo wrapper around the normal Gym CLI, not a new episode runner."""

import argparse
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen
from uuid import uuid4

import yaml


ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS = {"swe-pro": "swebench_pro", "tb21": "terminal_bench_2_1"}
HARNESSES = ("hermes", "pi")
# This is a rehearsed demo menu, not a declaration that every Cartesian pair works.
PAIRS = {("hermes", "swe-pro"), ("pi", "swe-pro"), ("pi", "tb21")}


def task_id(row: dict) -> str:
    """Show source task identity without changing the row sent to Gym."""
    return str(row.get("instance_id") or row.get("problem_id") or row.get("task_name") or "unknown")


def select_tasks(source: Path, *, limit: int, task_ids: list[str]) -> list[dict]:
    """Select unchanged prepared rows; never silently substitute a missing task."""
    rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    if task_ids:
        matches = {name: [row for row in rows if task_id(row) == name] for name in task_ids}
        if any(len(value) != 1 for value in matches.values()):
            raise ValueError("Each --task-id must match exactly one prepared row")
        rows = [matches[name][0] for name in task_ids]
    if not rows or len(rows) < limit:
        raise ValueError(f"Need {limit} prepared tasks; found {len(rows)}")
    return rows[:limit]


def composition(*, harness: str, benchmark: str, output: Path, head_port: int) -> dict:
    """Bind independent Agent and Resources configs through EnvironmentServer."""
    if (harness, benchmark) not in PAIRS:
        raise ValueError("Pair is outside this demo's validated menu")
    agent = f"{harness}_agent"
    resources = BENCHMARKS[benchmark]
    settings = {"num_workers": 1, "concurrency": 1}
    if harness == "pi":
        settings.update(timeout=1800, max_output_tokens=8192)
    else:
        settings.update(
            enabled_toolsets=["terminal"],
            max_turns=100,
            max_tokens=8192,
            temperature=0.0,
            sandbox_runner_timeout_seconds=1800,
        )
    return {
        "config_paths": [
            "examples/harness-swaps/common.yaml",
            f"resources_servers/{resources}/configs/{resources}.yaml",
            f"responses_api_agents/{agent}/configs/{agent}.yaml",
        ],
        "head_server": {"host": "127.0.0.1", "port": head_port},
        "results_dir": str(output),
        "nemo_gym_log_dir": str(output / "logs"),
        "model_call_capture_dir": str(output / "model-calls"),
        "single_agent_turn_legacy": {
            "environment_servers": {
                "single_agent_turn_legacy": {
                    "agent_server": {"name": agent},
                    "resources_server": {"name": f"{resources}_resources_server"},
                }
            }
        },
        agent: {"responses_api_agents": {agent: settings}},
        f"{resources}_resources_server": {
            "resources_servers": {
                resources: {
                    "num_workers": 1,
                    "sandbox_config": {"resources": {"cpu": 2, "memory_mib": 8192, "disk_gib": 30}},
                }
            }
        },
        "sandbox": {"docker": {"create": {"extra_run_args": ["--label", f"gym-swap-demo={output.name}"]}}},
    }


def summarize(output: Path, *, expected: int) -> dict:
    """Separate collection, verification and reward instead of equating exit 0 with success."""
    path = output / "rollouts.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []
    results = []
    for row in rows:
        response = row.get("response") or {}
        items = response.get("output") or []
        results.append(
            {
                "task": task_id(row),
                "reward": row.get("reward"),
                "evaluation_completed": row.get("evaluation_completed"),
                "mask_sample": row.get("mask_sample"),
                "tool_calls": sum(item.get("type") == "function_call" for item in items),
                "tool_results": sum(item.get("type") == "function_call_output" for item in items),
                "usage": response.get("usage"),
                "error": row.get("error") or row.get("error_message"),
            }
        )
    return {
        "expected": expected,
        "collected": len(rows),
        "results": results,
        "verification_complete": len(rows) == expected
        and all(
            row["evaluation_completed"] is True and not row["mask_sample"] and not row["error"] for row in results
        ),
        "health_report_present": (output / "quality_summary.json").exists(),
    }


def trace_progress(output: Path) -> tuple[int, int]:
    """Count actual captured model exchanges and tool requests, without displaying payloads."""
    calls = {}
    for path in (output / "model-calls").glob("**/*.jsonl"):
        with path.open() as stream:
            for line in stream:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    # The capture writer may be part-way through the final append.
                    continue
                if row.get("model_call_id"):
                    calls[row["model_call_id"]] = row
    tools = 0
    for row in calls.values():
        response = row.get("response") or {}
        for choice in response.get("choices") or []:
            tools += len((choice.get("message") or {}).get("tool_calls") or [])
    return len(calls), tools


def stop_process(proc: subprocess.Popen, *, grace: float = 60) -> None:
    """Signal only this invocation's session; give Gym a chance to close its children."""
    if proc.poll() is None:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=15)


def run_demo(args: argparse.Namespace) -> int:
    """Run one selected pair using CLI-managed services and canonical collection."""
    missing = [
        name
        for name in ("DEMO_MODEL_URL", "DEMO_MODEL_NAME", "DEMO_MODEL_KEY", "DEMO_GYM_HOST")
        if not os.environ.get(name)
    ]
    if missing:
        raise ValueError("Set private environment variables first: " + ", ".join(missing))
    rows = select_tasks(args.input.resolve(), limit=args.limit, task_ids=args.task_id)
    run_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{args.harness}-{args.benchmark}-{uuid4().hex[:6]}"
    output = args.output_root.resolve() / run_id
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    config = output / "run.yaml"
    config.write_text(
        yaml.safe_dump(
            composition(harness=args.harness, benchmark=args.benchmark, output=output, head_port=port), sort_keys=False
        )
    )
    inputs = output / "tasks.jsonl"
    inputs.write_text("".join(json.dumps(row) + "\n" for row in rows))
    started = time.monotonic()

    def report(message: str) -> None:
        # Only controlled progress messages enter the recording. Raw server/model logs stay private.
        print(message, flush=True)
        with (output / "events.jsonl").open("a") as stream:
            stream.write(json.dumps({"seconds": round(time.monotonic() - started, 3), "message": message}) + "\n")

    gym = str(Path(sys.executable).parent / "gym")
    start_cmd = [gym, "env", "start", "--config", str(config)]
    collect_cmd = [
        gym,
        "eval",
        "run",
        "--no-serve",
        "--config",
        str(config),
        "--agent",
        f"{args.harness}_agent",
        "--input",
        str(inputs),
        "--output",
        str(output / "rollouts.jsonl"),
        "--limit",
        str(args.limit),
        "--num-repeats",
        "1",
        "--concurrency",
        "1",
        "--health-check-workers",
        "1",
    ]
    env = os.environ | {"PYTHONUNBUFFERED": "1", "RAY_TMPDIR": "/tmp"}
    manifest = {
        "harness": args.harness,
        "benchmark": args.benchmark,
        "model": env["DEMO_MODEL_NAME"],
        "tasks": [task_id(row) for row in rows],
        "input_sha256": hashlib.sha256(inputs.read_bytes()).hexdigest(),
        "source_input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "commands": [start_cmd, collect_cmd],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    report(f"{args.harness.title()} × {args.benchmark} | {len(rows)} real task(s)")
    report("Model: " + manifest["model"])
    for row in rows:
        report("Task: " + str(row.get("repo") or task_id(row)))
    report("Artifacts: " + str(output.relative_to(ROOT) if output.is_relative_to(ROOT) else output))
    report("Preparing the run…")
    services = collector = None
    code = 1
    try:
        with (output / "services.log").open("w") as log:
            services = subprocess.Popen(
                start_cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
        deadline = time.monotonic() + 900
        while time.monotonic() < deadline:
            if services.poll() is not None:
                raise RuntimeError("Gym startup failed; inspect the private services.log")
            try:
                with urlopen(f"http://127.0.0.1:{port}/readyz", timeout=2) as response:
                    if response.status == 200:
                        break
            except (URLError, TimeoutError):
                pass
            time.sleep(2)
        else:
            raise TimeoutError("Gym startup timed out; inspect services.log")
        report("Ready. Running the selected task…")
        with (output / "collector.log").open("w") as log:
            collector = subprocess.Popen(
                collect_cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
        deadline = time.monotonic() + 4200
        last_progress = None
        while collector.poll() is None:
            if time.monotonic() >= deadline:
                raise TimeoutError("Collection deadline reached; retaining incomplete artifacts")
            progress = trace_progress(output)
            if progress != last_progress:
                report(f"Model calls: {progress[0]} | Tool requests: {progress[1]}")
                last_progress = progress
            time.sleep(5)
        code = collector.returncode
        result = summarize(output, expected=len(rows))
        result["collector_exit"] = code
        (output / "summary.json").write_text(json.dumps(result, indent=2))
        report(f"Collected {result['collected']}/{len(rows)} rollout(s); CLI exit {code}.")
        for row in result["results"]:
            report(
                f"Reward: {row['reward']} | verification completed: {row['evaluation_completed']} | "
                f"tool calls/results: {row['tool_calls']}/{row['tool_results']}"
            )
        report(f"Health report present: {result['health_report_present']} (see report for verdicts)")
        if not result["verification_complete"]:
            report("Verification is incomplete. Inspect saved logs; this is not a completed scored demo.")
            code = 1
    finally:
        if collector is not None:
            stop_process(collector)
        if services is not None:
            stop_process(services)
        inventory = subprocess.run(
            ["docker", "ps", "-aq", "--filter", f"label=gym-swap-demo={output.name}"],
            capture_output=True,
            text=True,
            errors="replace",
            check=True,
        )
        ids = inventory.stdout.split()
        # This label is unique to this invocation. Never prune or stop other containers.
        if ids:
            report(f"Removing {len(ids)} remaining demo-owned container(s) after shutdown.")
            subprocess.run(["docker", "rm", "-f", *ids], capture_output=True, check=True)
        (output / "cleanup.json").write_text(
            json.dumps({"containers_remaining_at_shutdown": ids, "demo_containers_removed": True}, indent=2)
        )
        report("Demo services stopped; demo-owned containers cleaned up.")
        report(f"Elapsed: {time.monotonic() - started:.0f} seconds (actual run time).")
    return code


def main() -> int:
    """Parse the demo-only selectors; every task still goes through Gym's CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--harness", choices=HARNESSES, required=True)
    parser.add_argument("--benchmark", choices=BENCHMARKS, required=True)
    parser.add_argument("--input", type=Path, help="Prepared JSONL; defaults to this demo's selected benchmark data")
    parser.add_argument("--task-id", action="append", default=[])
    parser.add_argument("--limit", type=int, choices=(1, 2), default=1)
    parser.add_argument("--output-root", type=Path, default=ROOT / "results/harness-swaps")
    args = parser.parse_args()
    if args.input is None:
        args.input = Path(__file__).parent / "data" / f"{args.benchmark}.jsonl"
    if (args.harness, args.benchmark) not in PAIRS:
        parser.error("This demo includes Hermes/SWE-Pro, Pi/SWE-Pro and Pi/TB2.1 only")
    os.umask(0o077)
    return run_demo(args)


if __name__ == "__main__":
    raise SystemExit(main())
