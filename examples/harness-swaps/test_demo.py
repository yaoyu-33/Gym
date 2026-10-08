# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test actual demo selection, composition and result interpretation."""

import importlib.util
import json
from pathlib import Path

import pytest
import yaml


spec = importlib.util.spec_from_file_location("swap_demo", Path(__file__).with_name("run.py"))
demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)


@pytest.mark.parametrize("harness,benchmark", sorted(demo.PAIRS))
def test_composition_routes_independent_parts(tmp_path, harness, benchmark):
    config = demo.composition(harness=harness, benchmark=benchmark, output=tmp_path, head_port=12345)
    route = config["single_agent_turn_legacy"]["environment_servers"]["single_agent_turn_legacy"]
    assert route["agent_server"]["name"] == f"{harness}_agent"
    resources = demo.BENCHMARKS[benchmark]
    assert route["resources_server"]["name"] == f"{resources}_resources_server"
    assert config["config_paths"][1] == f"resources_servers/{resources}/configs/{resources}.yaml"
    assert config["config_paths"][2].startswith(f"responses_api_agents/{harness}_agent/")
    assert config["model_call_capture_dir"] == str(tmp_path / "model-calls")
    assert config["sandbox"]["docker"]["create"]["extra_run_args"] == ["--label", f"gym-swap-demo={tmp_path.name}"]
    assert all((demo.ROOT / path).is_file() for path in config["config_paths"])


def test_unrehearsed_pair_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="outside"):
        demo.composition(harness="hermes", benchmark="tb21", output=tmp_path, head_port=12345)


def test_concurrency_has_required_queue_deadline():
    config = yaml.safe_load(Path(__file__).with_name("common.yaml").read_text())
    env = config["single_agent_turn_legacy"]["environment_servers"]["single_agent_turn_legacy"]
    assert env["max_concurrent_episodes"] == 1
    assert env["queue_timeout_seconds"] > 0


def test_selection_preserves_task_prompt_and_verifier(tmp_path):
    source = tmp_path / "input.jsonl"
    rows = [
        {
            "instance_id": str(i),
            "responses_create_params": {"input": "Unchanged prompt"},
            "patch": "verifier-only bytes\n",
            "test_patch": "original\n",
        }
        for i in range(3)
    ]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    original = source.read_bytes()
    assert demo.select_tasks(source, limit=2, task_ids=["2", "0"]) == [rows[2], rows[0]]
    assert source.read_bytes() == original
    with pytest.raises(ValueError, match="exactly one"):
        demo.select_tasks(source, limit=1, task_ids=["missing"])


def test_selection_rejects_short_input(tmp_path):
    source = tmp_path / "input.jsonl"
    source.write_text("{}\n")
    with pytest.raises(ValueError, match="Need 2"):
        demo.select_tasks(source, limit=2, task_ids=[])


def test_summary_keeps_incomplete_zero_distinct_from_completed_zero(tmp_path):
    rows = [
        {"reward": 0, "evaluation_completed": False, "error": "verifier unavailable"},
        {
            "reward": 0,
            "evaluation_completed": True,
            "response": {"output": [{"type": "function_call"}, {"type": "function_call_output"}]},
        },
    ]
    (tmp_path / "rollouts.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    result = demo.summarize(tmp_path, expected=2)
    assert result["collected"] == 2
    assert result["results"][0]["evaluation_completed"] is False
    assert result["results"][0]["error"] == "verifier unavailable"
    assert result["results"][1]["evaluation_completed"] is True
    assert result["results"][1]["tool_calls"] == result["results"][1]["tool_results"] == 1
    assert result["health_report_present"] is False
    assert result["verification_complete"] is False


def test_missing_rollouts_are_not_success(tmp_path):
    result = demo.summarize(tmp_path, expected=1)
    assert result["expected"] == 1
    assert result["collected"] == 0
    assert result["results"] == []
    assert result["verification_complete"] is False


def test_completed_zero_is_a_valid_result(tmp_path):
    (tmp_path / "rollouts.jsonl").write_text(json.dumps({"reward": 0, "evaluation_completed": True}) + "\n")
    assert demo.summarize(tmp_path, expected=1)["verification_complete"] is True


def test_progress_counts_unique_exchanges_and_ignores_partial_append(tmp_path):
    captures = tmp_path / "model-calls"
    captures.mkdir()
    exchange = {
        "model_call_id": "call-1",
        "response": {"choices": [{"message": {"tool_calls": [{"function": {"name": "terminal"}}]}}]},
    }
    (captures / "task.capture.jsonl").write_text(json.dumps(exchange) + "\n" + json.dumps(exchange) + '\n{"model')
    assert demo.trace_progress(tmp_path) == (1, 1)
