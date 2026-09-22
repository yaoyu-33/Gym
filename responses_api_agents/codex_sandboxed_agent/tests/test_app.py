# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from nemo_gym.base_responses_api_agent import AgentSeedSessionRequest
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.sandboxed_agent import SandboxedAgentSession
from nemo_gym.server_utils import ServerClient
from responses_api_agents.codex_sandboxed_agent.app import CodexSandboxedAgent, CodexSandboxedAgentConfig


@pytest.fixture
def agent():
    client = MagicMock(spec=ServerClient)
    client.global_config_dict = {"policy": {"responses_api_models": {"vllm_model": {"host": "model", "port": 8001}}}}
    client._build_server_base_url.return_value = "http://model:8001"
    return CodexSandboxedAgent(
        config=CodexSandboxedAgentConfig(
            name="codex",
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
async def test_nonzero_cli_exit_preserves_tool_progress_without_scoring(agent, session, monkeypatch):
    stdout = "\n".join(
        [
            json.dumps(
                {
                    "type": "item.completed",
                    "item": {
                        "type": "command_execution",
                        "id": "tool",
                        "command": "git diff",
                        "aggregated_output": "actual patch",
                        "exit_code": 0,
                        "status": "completed",
                    },
                }
            ),
            json.dumps({"type": "turn.failed", "error": {"message": "model unavailable"}}),
        ]
    )
    monkeypatch.setattr(type(agent), "upload_text", AsyncMock())
    monkeypatch.setattr(
        type(agent), "run_cli", AsyncMock(return_value=({"return_code": 1, "timed_out": False}, stdout))
    )
    response = await agent.execute_response(session, None, NeMoGymResponseCreateParamsNonStreaming(input="fix"))
    assert response.status == "failed"
    assert [item.type for item in response.output] == ["function_call", "function_call_output"]
    assert "actual patch" in response.output[1].output
    assert "reward" not in response.model_dump()
