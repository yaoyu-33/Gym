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
from responses_api_agents.pi_sandboxed_agent.app import PiSandboxedAgent, PiSandboxedAgentConfig


@pytest.fixture
def agent():
    client = MagicMock(spec=ServerClient)
    client.global_config_dict = {"policy": {"responses_api_models": {"vllm_model": {"host": "model", "port": 8001}}}}
    client._build_server_base_url.return_value = "http://model:8001"
    return PiSandboxedAgent(
        config=PiSandboxedAgentConfig(
            name="pi",
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
@pytest.mark.parametrize("stop_reason", ["stop", "error", "length"])
async def test_preserves_reasoning_usage_and_model_error(agent, session, monkeypatch, stop_reason):
    event = {
        "type": "message_end",
        "message": {
            "role": "assistant",
            "content": [{"type": "thinking", "thinking": "inspect the bug"}],
            "usage": {"input": 7, "cacheRead": 3, "output": 4},
            "stopReason": stop_reason,
            "errorMessage": "quota",
        },
    }
    monkeypatch.setattr(type(agent), "upload_text", AsyncMock())
    monkeypatch.setattr(
        type(agent), "run_cli", AsyncMock(return_value=({"return_code": 0, "timed_out": False}, json.dumps(event)))
    )
    response = await agent.execute_response(session, None, NeMoGymResponseCreateParamsNonStreaming(input="fix"))
    assert response.status == {"stop": "completed", "error": "failed", "length": "incomplete"}[stop_reason]
    assert response.output[0].type == "reasoning"
    assert response.output[0].summary[0].text == "inspect the bug"
    assert response.usage.input_tokens == 10 and response.usage.input_tokens_details.cached_tokens == 3
    assert session.observations.records[0].conversation[-1].type == "reasoning"
