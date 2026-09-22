# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib
import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml

from nemo_gym.base_responses_api_agent import AgentSeedSessionRequest
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.sandbox import SandboxExecResult
from nemo_gym.sandboxed_agent import SandboxedAgentSession
from nemo_gym.sandboxed_cli_agent import text_prompt
from nemo_gym.server_utils import ServerClient


@pytest.fixture(params=[("pi", "Pi"), ("openclaw", "OpenClaw"), ("codex", "Codex")])
def agent(request):
    name, prefix = request.param
    module = importlib.import_module(f"responses_api_agents.{name}_sandboxed_agent.app")
    config = getattr(module, prefix + "SandboxedAgentConfig")(
        name=name,
        host="127.0.0.1",
        port=8000,
        entrypoint="app.py",
        model="super",
        model_server={"type": "responses_api_models", "name": "policy"},
    )
    client = MagicMock(spec=ServerClient)
    client.global_config_dict = {"policy": {"responses_api_models": {"vllm_model": {"host": "model", "port": 8001}}}}
    client._build_server_base_url.return_value = "http://model:8001"
    return getattr(module, prefix + "SandboxedAgent")(config=config, server_client=client)


@pytest.fixture
def session():
    seed = AgentSeedSessionRequest(
        episode_id={"rollout_id": "task", "attempt": 1}, task_id={"taskset": "pro", "task_id": "bug"}
    )
    box = SimpleNamespace(upload=AsyncMock(), disconnect=AsyncMock(), stop=AsyncMock())
    return SandboxedAgentSession(seed, box, "/tmp/session", "/app")


def test_instructions_and_history():
    body = NeMoGymResponseCreateParamsNonStreaming(
        input=[{"role": "system", "content": "system"}, {"role": "user", "content": "fix"}], instructions="request"
    )
    assert text_prompt(body, "config") == ("fix", "config\n\nrequest\n\nsystem")
    with pytest.raises(ValueError, match="single user turn"):
        text_prompt(NeMoGymResponseCreateParamsNonStreaming(input=[{"role": "assistant", "content": "prior"}]), None)


@pytest.mark.parametrize(
    "fields",
    [
        {"top_p": 0.2},
        {"reasoning": {"effort": "high"}},
        {"previous_response_id": "old"},
        {"tool_choice": "none"},
        {"parallel_tool_calls": False},
        {"metadata": {"chat_template_kwargs": "{}"}},
    ],
)
def test_unsupported_controls_are_not_silently_dropped(fields):
    with pytest.raises(ValueError):
        text_prompt(NeMoGymResponseCreateParamsNonStreaming(input="fix", **fields), None)


@pytest.mark.asyncio
async def test_preflight_rejects_wrong_pin_before_upload(agent, session, monkeypatch):
    execute = AsyncMock(return_value=SandboxExecResult(return_code=0, stdout="version 0.0.0", stderr=""))
    monkeypatch.setattr(type(agent), "exec_in_session", execute)
    with pytest.raises(RuntimeError, match="Prepare"):
        await agent.prepare_session(session)
    session.sandbox.upload.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("ack", [None, False])
async def test_unconfirmed_runner_cleanup_blocks_close(agent, session, monkeypatch, ack):
    monkeypatch.setattr(type(agent), "upload_text", AsyncMock())
    monkeypatch.setattr(
        type(agent), "exec_in_session", AsyncMock(return_value=SandboxExecResult(return_code=0, stdout="", stderr=""))
    )
    monkeypatch.setattr(type(agent), "download_json", AsyncMock(return_value={"cleanup_confirmed": ack}))
    with pytest.raises(RuntimeError, match="confirm cleanup"):
        await agent.run_cli(session, [agent.cli], env={})
    with pytest.raises(RuntimeError, match="termination is unconfirmed"):
        await agent._close(session)
    session.sandbox.disconnect.assert_not_awaited()
    session.sandbox.stop.assert_not_awaited()


@pytest.mark.asyncio
async def test_runner_uses_borrowed_workdir_and_isolated_home(agent, session, monkeypatch):
    upload, execute = AsyncMock(), AsyncMock(return_value=SandboxExecResult(return_code=0, stdout="", stderr=""))
    monkeypatch.setattr(type(agent), "upload_text", upload)
    monkeypatch.setattr(type(agent), "exec_in_session", execute)
    monkeypatch.setattr(
        type(agent), "download_json", AsyncMock(return_value={"cleanup_confirmed": True, "return_code": 0})
    )
    monkeypatch.setattr(type(agent), "download_text", AsyncMock(return_value="actual output"))
    _, stdout = await agent.run_cli(session, [agent.cli, "solve"], env={"OPENAI_API_KEY": "gym"})
    payload = json.loads(upload.await_args.args[2])
    assert payload["cwd"] == "/app" and payload["env"]["HOME"] == "/tmp/session/home"
    assert "PATH" not in payload["env"]
    assert payload["command"][0].startswith("/opt/gym-cli/")
    assert execute.await_args.kwargs["timeout_s"] > payload["timeout"]
    assert stdout == "actual output"
    session.sandbox.stop.assert_not_awaited()


@pytest.mark.asyncio
async def test_native_execution_route_output_and_observations(agent, session, monkeypatch):
    completed = {"cleanup_confirmed": True, "return_code": 0, "timed_out": False}
    message = {
        "role": "assistant",
        "content": [{"type": "text", "text": "patched"}],
        "usage": {"input": 10, "output": 2},
        "stopReason": "stop",
    }
    if agent.cli_name == "pi":
        stdout = json.dumps({"type": "message_end", "message": message})
    elif agent.cli_name == "codex":
        stdout = "\n".join(
            [
                json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": "patched"}}),
                json.dumps({"type": "turn.completed", "usage": {"input_tokens": 10, "output_tokens": 2}}),
            ]
        )
    else:
        stdout = json.dumps({"meta": {"agentMeta": {"sessionFile": "/tmp/session/home/run.jsonl"}}})
    execute, upload = AsyncMock(return_value=(completed, stdout)), AsyncMock()
    monkeypatch.setattr(type(agent), "run_cli", execute)
    monkeypatch.setattr(type(agent), "upload_text", upload)
    monkeypatch.setattr(type(agent), "download_json", AsyncMock(return_value={"agents": {"list": [{"id": "main"}]}}))
    monkeypatch.setattr(
        type(agent), "download_text", AsyncMock(return_value=json.dumps({"type": "message", "message": message}))
    )
    response = await agent.execute_response(
        session, None, NeMoGymResponseCreateParamsNonStreaming(input="fix", instructions="be precise")
    )
    assert response.status == "completed" and response.output[-1].content[0].text == "patched"
    assert session.observations.records[0].conversation[-1] == response.output[-1]
    assert "reward" not in response.model_dump()
    config_text = upload.await_args.args[2]
    assert "http://model:8001/ng-rollout/task-a1/v1" in config_text
    if agent.cli_name == "codex":
        config = tomllib.loads(config_text)
        assert config["model_providers"]["gym"]["wire_api"] == "responses"
        assert config["developer_instructions"] == "be precise"
        assert execute.await_args.args[1][-2:] == ["--", "fix"]
    elif agent.cli_name == "openclaw":
        config = json.loads(config_text)
        assert config["agents"]["defaults"]["workspace"] == "/app"
        assert config["agents"]["list"][0]["workspace"] == "/app"
        assert config["models"]["providers"]["nemo"]["api"] == "openai-completions"
    else:
        assert json.loads(config_text)["providers"]["nemo"]["api"] == "openai-completions"
        assert "be precise" in execute.await_args.args[1]
    agent.server_client.post.assert_not_called()


def test_native_yaml_references_real_resources(agent):
    from environment_servers.single_agent.app import SingleAgentEnvironmentServerConfig

    root, name = Path(__file__).resolve().parents[2], agent.cli_name
    config = yaml.safe_load(
        (root / f"responses_api_agents/{name}_sandboxed_agent/configs/{name}_sandboxed_agent.yaml").read_text()
    )
    assert (root / config["config_paths"][0]).is_file()
    env = config[f"swe_pro_{name}"]["environment_servers"]["single_agent"]
    env = SingleAgentEnvironmentServerConfig(name="env", host="127.0.0.1", port=8000, **env)
    assert env.agent_server.name == f"{name}_sandboxed_agent"
    assert env.resources_server.name == "swebench_pro_resources_server"


@pytest.mark.skipif(sys.platform != "linux", reason="Real Linux subreaper and /proc are required")
@pytest.mark.parametrize("timeout", [0.1, 5])
def test_standalone_runner_reaps_detached_descendants(tmp_path, timeout):
    from nemo_gym import sandboxed_cli_runner

    code = (
        "import subprocess,sys,pathlib,time; p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(20)'],"
        "start_new_session=True); pathlib.Path('child.pid').write_text(str(p.pid)); "
        + ("time.sleep(20)" if timeout < 1 else "time.sleep(0.02)")
    )
    params = {
        "command": [sys.executable, "-c", code],
        "cwd": str(tmp_path),
        "directory": str(tmp_path),
        "env": {},
        "timeout": timeout,
        "cleanup_timeout": 2,
    }
    request = tmp_path / "command.json"
    request.write_text(json.dumps(params))
    try:
        subprocess.run([sys.executable, "-I", sandboxed_cli_runner.__file__, str(request)], timeout=10, check=True)
        result = json.loads((tmp_path / "result.json").read_text())
        assert result["cleanup_confirmed"] is True and result["timed_out"] is (timeout < 1)
        with pytest.raises(ProcessLookupError):
            os.kill(int((tmp_path / "child.pid").read_text()), 0)
    finally:
        if (tmp_path / "child.pid").exists():
            try:
                os.kill(int((tmp_path / "child.pid").read_text()), 9)
            except ProcessLookupError:
                pass
