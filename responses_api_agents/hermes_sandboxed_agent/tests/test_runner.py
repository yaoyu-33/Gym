# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import sys
from types import ModuleType, SimpleNamespace

import pytest

from responses_api_agents.hermes_sandboxed_agent import runner


@pytest.mark.parametrize("leftover", [False, True])
def test_runner_snapshots_before_close_and_checks_owned_background_jobs(tmp_path, monkeypatch, leftover):
    calls = []
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / "hermes-runtime.json").write_text(json.dumps({"hermes_commit": "pinned"}))
    monkeypatch.setattr(sys, "prefix", str(runtime))
    monkeypatch.setattr(os, "environ", dict(os.environ))
    monkeypatch.setattr(runner.subprocess, "check_output", lambda *args, **kwargs: "ripgrep 14.1.1\n")
    monkeypatch.setattr(runner.signal, "signal", lambda *args: None)
    monkeypatch.setattr(
        runner,
        "Timer",
        lambda *args: SimpleNamespace(
            start=lambda: calls.append("timer-start"), cancel=lambda: calls.append("timer-cancel")
        ),
    )
    module = ModuleType("run_agent")
    module.__file__ = str(runtime / "hermes-src/run_agent.py")

    class AIAgent:
        def __init__(self, **kwargs):
            assert kwargs["api_mode"] == "chat_completions"
            assert kwargs["api_key"] == "gym"
            assert kwargs["max_tokens"] == 128
            assert kwargs["request_overrides"]["temperature"] == 0
            self.messages = [{"role": "user", "content": "fix"}, {"role": "assistant", "content": "patched"}]
            self.session_input_tokens, self.session_output_tokens = 10, 20
            self.session_cache_read_tokens, self.session_reasoning_tokens = 0, 5
            self.iteration_budget = SimpleNamespace(remaining=89)
            self.context_compressor = SimpleNamespace(context_length=1000, threshold_tokens=850)
            self._process_owner_task_ids = {"our-task"}

        def run_conversation(self, query, system, history):
            assert query == "fix"
            return {"completed": True, "api_calls": 1, "messages": self.messages}

        def close(self):
            calls.append("agent-close")
            self.messages.clear()

    module.AIAgent = AIAgent
    state = ModuleType("hermes_state")
    state.SessionDB = lambda: SimpleNamespace(close=lambda: calls.append("database-close"))
    registry = ModuleType("tools.process_registry")
    registry.process_registry = SimpleNamespace(
        list_sessions=lambda: [
            {"owner_task_id": "another-task", "status": "running"},
            {"owner_task_id": "our-task", "status": "running" if leftover else "exited"},
        ]
    )
    monkeypatch.setitem(sys.modules, "run_agent", module)
    monkeypatch.setitem(sys.modules, "hermes_state", state)
    monkeypatch.setitem(sys.modules, "tools.process_registry", registry)
    result = runner.run(
        {
            "run_dir": str(tmp_path / "session"),
            "workdir": "/app",
            "model": "super",
            "base_url": "http://gym/v1",
            "max_turns": 90,
            "max_tokens": 128,
            "temperature": 0,
            "wall_time": 2700,
            "terminal_timeout": 180,
            "api_timeout": 1800,
            "compression_enabled": True,
            "chat_template_kwargs": {},
            "enabled_toolsets": None,
            "disabled_toolsets": None,
            "input": "fix",
            "system_prompt": None,
        }
    )
    assert calls.index("agent-close") < calls.index("database-close")
    assert result["messages"][-1]["content"] == "patched"
    assert result["usage"]["output_tokens"] == 20
    assert result["cleanup_confirmed"] is not leftover
    assert result["runtime"]["tool_cwd"] == "/app"


def test_reasoning_budget_is_not_a_harness_crash():
    result = runner.classify_stop(
        {
            "partial": True,
            "error": "Model used all output tokens on reasoning with none left for the response.",
            "messages": [{"role": "assistant", "reasoning_content": "thinking"}],
        }
    )
    assert result["stop_reason"] == "output_tokens"
    assert result["budget_exhausted"] is True
    assert not result["failed"]
