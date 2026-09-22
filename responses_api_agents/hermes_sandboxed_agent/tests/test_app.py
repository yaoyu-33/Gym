# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from nemo_gym.base_responses_api_agent import AgentSeedSessionRequest
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.sandbox import SandboxExecResult
from nemo_gym.sandboxed_agent import SandboxedAgentSession
from nemo_gym.server_utils import ServerClient
from responses_api_agents.hermes_sandboxed_agent.app import HermesSandboxedAgent, HermesSandboxedAgentConfig
from responses_api_agents.hermes_sandboxed_agent.trajectory import trajectory_response


@pytest.fixture
def agent():
    client = MagicMock(spec=ServerClient)
    client.global_config_dict = {"policy": {"responses_api_models": {"vllm_model": {"host": "model", "port": 8001}}}}
    client._build_server_base_url.return_value = "http://model:8001"
    return HermesSandboxedAgent(
        config=HermesSandboxedAgentConfig(
            name="hermes",
            host="127.0.0.1",
            port=8000,
            entrypoint="app.py",
            model="super",
            model_server={"type": "responses_api_models", "name": "policy"},
            temperature=1.0,
            max_tokens=4096,
        ),
        server_client=client,
    )


@pytest.fixture
def session():
    seed = AgentSeedSessionRequest(
        episode_id={"rollout_id": "sample", "attempt": 1}, task_id={"taskset": "pro", "task_id": "bug"}
    )
    return SandboxedAgentSession(seed, SimpleNamespace(upload=AsyncMock()), "/tmp/agent-123", "/app")


@pytest.mark.parametrize("temperature,max_tokens", [(0, 128), (0.3, 8192)])
def test_request_sampling_overrides_and_no_task_metadata(agent, session, temperature, max_tokens):
    params = agent.request_parameters(
        session,
        None,
        NeMoGymResponseCreateParamsNonStreaming(
            input="fix the bug",
            temperature=temperature,
            max_output_tokens=max_tokens,
            metadata={"chat_template_kwargs": '{"enable_thinking": false}'},
        ),
    )
    assert params["temperature"] == temperature
    assert params["max_tokens"] == max_tokens
    assert params["base_url"] == "http://model:8001/ng-rollout/sample-a1/v1"
    assert params["workdir"] == "/app"
    assert params["chat_template_kwargs"] == {"enable_thinking": False}
    assert "task_data" not in params and "patch" not in params


def test_sampling_falls_back_to_config(agent, session):
    params = agent.request_parameters(session, None, NeMoGymResponseCreateParamsNonStreaming(input="fix"))
    assert params["temperature"] == 1.0
    assert params["max_tokens"] == 4096


@pytest.mark.asyncio
@pytest.mark.parametrize("marker", [None, False, True])
async def test_only_explicit_cleanup_acknowledgement_allows_verification(agent, session, monkeypatch, marker):
    result = {"completed": True, "messages": [{"role": "assistant", "content": "patched"}], "n_input": 0}
    if marker is not None:
        result["cleanup_confirmed"] = marker
    monkeypatch.setattr(type(agent), "upload_text", AsyncMock())
    monkeypatch.setattr(
        type(agent), "exec_in_session", AsyncMock(return_value=SandboxExecResult(return_code=0, stdout="", stderr=""))
    )
    monkeypatch.setattr(type(agent), "download_json", AsyncMock(return_value=result))
    await agent.execute_response(session, None, NeMoGymResponseCreateParamsNonStreaming(input="fix"))
    assert session.execution_uncertain is (marker is not True)


@pytest.mark.asyncio
async def test_seed_checks_runtime_before_upload(agent, session, monkeypatch):
    execute = AsyncMock(return_value=SandboxExecResult(return_code=1, stdout="", stderr="wrong pin"))
    monkeypatch.setattr(type(agent), "exec_in_session", execute)
    with pytest.raises(RuntimeError, match="pinned Hermes"):
        await agent.prepare_session(session)
    assert agent.config.hermes_commit in execute.await_args.args[1]
    session.sandbox.upload.assert_not_awaited()


@pytest.mark.parametrize("failed", [False, True])
def test_trajectory_keeps_tools_reasoning_and_actual_tokens(failed):
    response = trajectory_response(
        {
            "completed": not failed,
            "failed": failed,
            "error": "harness error" if failed else None,
            "n_input": 1,
            "messages": [
                {"role": "user", "content": "bug"},
                {
                    "role": "assistant",
                    "reasoning_content": "inspect",
                    "tool_calls": [{"id": "tool", "function": {"name": "terminal", "arguments": "{}"}}],
                },
                {"role": "tool", "tool_call_id": "tool", "content": "patch applied"},
                {"role": "assistant", "content": "fixed"},
            ],
            "usage": {"input_tokens": 120, "output_tokens": 30, "cached_tokens": 10, "reasoning_tokens": 5},
        },
        NeMoGymResponseCreateParamsNonStreaming(input="bug"),
        "super",
    )
    assert [item.type for item in response.output] == ["reasoning", "function_call", "function_call_output", "message"]
    assert response.status == ("failed" if failed else "completed")
    assert response.usage.total_tokens == 150
    assert response.usage.input_tokens_details.cached_tokens == 10
    assert response.output[1].call_id == response.output[2].call_id
    assert "reward" not in response.model_dump()
