# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from nemo_gym.base_responses_api_agent import AgentSeedSessionRequest
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.sandbox import SandboxExecResult
from nemo_gym.sandboxed_agent import SandboxedAgentSession
from nemo_gym.server_utils import ServerClient
from responses_api_agents.openclaw_sandboxed_agent.app import OpenClawSandboxedAgent, OpenClawSandboxedAgentConfig


@pytest.fixture
def agent():
    client = MagicMock(spec=ServerClient)
    client.global_config_dict = {"policy": {"responses_api_models": {"vllm_model": {"host": "model", "port": 8001}}}}
    client._build_server_base_url.return_value = "http://model:8001"
    return OpenClawSandboxedAgent(
        config=OpenClawSandboxedAgentConfig(
            name="openclaw",
            host="127.0.0.1",
            port=8000,
            entrypoint="app.py",
            model="super",
            model_server={"type": "responses_api_models", "name": "policy"},
        ),
        server_client=client,
    )


@pytest.fixture
def session():
    seed = AgentSeedSessionRequest(episode_id={"rollout_id": "task"}, task_id={"taskset": "pro", "task_id": "bug"})
    return SandboxedAgentSession(seed, SimpleNamespace(), "/tmp/session", "/app")


@pytest.mark.asyncio
async def test_timeout_salvages_partial_native_transcript(agent, session, monkeypatch):
    calls = AsyncMock(
        side_effect=[
            ({"return_code": 0, "timed_out": False}, ""),
            ({"return_code": -9, "timed_out": True}, "partial non-JSON logs"),
        ]
    )
    monkeypatch.setattr(type(agent), "run_cli", calls)
    monkeypatch.setattr(type(agent), "upload_text", AsyncMock())
    monkeypatch.setattr(type(agent), "download_json", AsyncMock(return_value={}))
    monkeypatch.setattr(
        type(agent),
        "exec_in_session",
        AsyncMock(
            return_value=SandboxExecResult(
                return_code=0, stdout=json.dumps(["/tmp/session/home/partial.jsonl"]), stderr=""
            )
        ),
    )
    transcript = json.dumps(
        {
            "type": "message",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "partial reasoning"},
                    {"type": "toolCall", "id": "call", "name": "exec", "arguments": {"command": "git diff"}},
                ],
                "usage": {"input": 10, "output": 5},
            },
        }
    )
    monkeypatch.setattr(type(agent), "download_text", AsyncMock(return_value=transcript))
    response = await agent.execute_response(session, None, NeMoGymResponseCreateParamsNonStreaming(input="fix"))
    assert response.status == "incomplete"
    assert [item.type for item in response.output] == ["reasoning", "function_call"]
    assert response.usage.total_tokens == 15
    command = calls.await_args.args[1]
    assert command[command.index("--timeout") + 1] == str(int(agent.config.sandbox_timeout))
    assert session.observations.records[0].conversation[-1].name == "exec"


@pytest.mark.asyncio
async def test_native_error_with_zero_exit_is_not_success(agent, session, monkeypatch):
    envelope = {
        "meta": {
            "error": {"kind": "context_overflow"},
            "agentMeta": {"sessionFile": "/tmp/session/home/partial.jsonl"},
        }
    }
    monkeypatch.setattr(
        type(agent), "run_cli", AsyncMock(return_value=({"return_code": 0, "timed_out": False}, json.dumps(envelope)))
    )
    monkeypatch.setattr(type(agent), "upload_text", AsyncMock())
    monkeypatch.setattr(type(agent), "download_json", AsyncMock(return_value={}))
    monkeypatch.setattr(
        type(agent),
        "download_text",
        AsyncMock(
            return_value=json.dumps(
                {"type": "message", "message": {"role": "assistant", "content": [{"type": "text", "text": "partial"}]}}
            )
        ),
    )
    response = await agent.execute_response(session, None, NeMoGymResponseCreateParamsNonStreaming(input="fix"))
    assert response.status == "failed"
    assert response.output[0].content[0].text == "partial"
    assert response.metadata["budget_exhausted"] == "false"
