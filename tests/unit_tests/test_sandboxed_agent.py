# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from nemo_gym.base_responses_api_agent import AgentCloseSessionRequest, AgentSeedSessionRequest
from nemo_gym.episode_types import EpisodeId, TaskId
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.sandbox import SandboxExecResult
from nemo_gym.sandboxed_agent import SandboxedAgentConfig, SandboxedResponsesAPIAgent
from nemo_gym.server_utils import ServerClient


class ExampleAgent(SandboxedResponsesAPIAgent):
    observation_source = "example"

    async def prepare_session(self, session):
        pass

    async def execute_response(self, session, request, body):
        result = await self.exec_in_session(session, "solve", timeout_s=5)
        return NeMoGymResponse.model_validate(
            {
                "id": "response",
                "object": "response",
                "created_at": 0,
                "model": "policy",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "id": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": result.stdout, "annotations": []}],
                    }
                ],
                "parallel_tool_calls": False,
                "tool_choice": "auto",
                "tools": [],
            }
        )


def make_agent(**kwargs):
    return ExampleAgent(
        config=SandboxedAgentConfig(
            name="example",
            host="127.0.0.1",
            port=8000,
            entrypoint="app.py",
            model="policy",
            model_server={"type": "responses_api_models", "name": "policy"},
            **kwargs,
        ),
        server_client=MagicMock(spec=ServerClient),
    )


def seed_request():
    return AgentSeedSessionRequest(
        episode_id=EpisodeId(rollout_id="rollout", attempt=2),
        task_id=TaskId(taskset="swe", task_id="task"),
        sandbox_access={
            "workdir": "/app",
            "connection": {
                "kind": "direct",
                "provider_config_ref": "sandbox",
                "descriptor": {"sandbox_id": "owned-by-resources"},
            },
        },
    )


@pytest.fixture
def sandbox(monkeypatch):
    box = SimpleNamespace(
        exec=AsyncMock(return_value=SandboxExecResult(stdout="fixed", stderr="", return_code=0)),
        disconnect=AsyncMock(),
        stop=AsyncMock(),
    )
    monkeypatch.setattr("nemo_gym.sandboxed_agent.get_global_config_dict", lambda: {})
    monkeypatch.setattr("nemo_gym.sandboxed_agent.resolve_provider_config", lambda *args: {})
    monkeypatch.setattr("nemo_gym.sandboxed_agent.create_provider", lambda *args: SimpleNamespace(aclose=AsyncMock()))
    monkeypatch.setattr("nemo_gym.sandboxed_agent.AsyncSandbox.connect", AsyncMock(return_value=box))
    return box


@pytest.mark.asyncio
async def test_seed_activate_close_uses_exact_borrowed_workdir_and_preserves_output(sandbox):
    agent = make_agent()
    request = SimpleNamespace(session={}, path_params={"rollout_id": "rollout-a2"})
    opened = await agent.seed_agent_session(request, seed_request())
    response = await agent.responses(request, NeMoGymResponseCreateParamsNonStreaming(input="fix"))
    assert response.output[0].content[0].text == "fixed"
    assert sandbox.exec.await_args.kwargs["cwd"] == "/app"
    closed = await agent.close_agent_session(
        request,
        AgentCloseSessionRequest(agent_session_id=opened.agent_session_id, episode_id=seed_request().episode_id),
    )
    assert closed.agent_observations.records[0].conversation == response.output
    sandbox.disconnect.assert_awaited_once()
    sandbox.stop.assert_not_awaited()
    assert not agent._sessions
    agent.server_client.post.assert_not_called()
    replay = await agent.close_agent_session(
        request,
        AgentCloseSessionRequest(agent_session_id=opened.agent_session_id, episode_id=seed_request().episode_id),
    )
    assert replay == closed
    sandbox.disconnect.assert_awaited_once()


@pytest.mark.asyncio
async def test_identity_single_activation_and_no_local_fallback(sandbox):
    agent = make_agent()
    request = SimpleNamespace(session={}, path_params={"rollout_id": "wrong"})
    with pytest.raises(HTTPException) as error:
        await agent.responses(request, NeMoGymResponseCreateParamsNonStreaming(input="fix"))
    assert error.value.status_code == 409
    missing = seed_request().model_copy(update={"sandbox_access": None})
    with pytest.raises(HTTPException) as error:
        await agent.seed_agent_session(request, missing)
    assert error.value.status_code == 422
    opened = await agent.seed_agent_session(request, seed_request())
    with pytest.raises(HTTPException):
        await agent.seed_agent_session(request, seed_request())
    with pytest.raises(HTTPException):
        await agent.responses(request, NeMoGymResponseCreateParamsNonStreaming(input="fix"))
    sandbox.exec.assert_not_awaited()
    request.path_params["rollout_id"] = "rollout-a2"
    await agent.responses(request, NeMoGymResponseCreateParamsNonStreaming(input="fix"))
    with pytest.raises(HTTPException):
        await agent.responses(request, NeMoGymResponseCreateParamsNonStreaming(input="fix"))
    with pytest.raises(HTTPException):
        await agent.close_agent_session(
            request,
            AgentCloseSessionRequest(
                agent_session_id=opened.agent_session_id, episode_id=EpisodeId(rollout_id="other")
            ),
        )
    sandbox.disconnect.assert_not_awaited()
    await agent.close_agent_session(
        request,
        AgentCloseSessionRequest(agent_session_id=opened.agent_session_id, episode_id=seed_request().episode_id),
    )


@pytest.mark.asyncio
async def test_disconnect_failure_is_retryable_without_losing_state(sandbox):
    agent = make_agent()
    request = SimpleNamespace(session={}, path_params={})
    opened = await agent.seed_agent_session(request, seed_request())
    close = AgentCloseSessionRequest(agent_session_id=opened.agent_session_id, episode_id=seed_request().episode_id)
    sandbox.disconnect.side_effect = [RuntimeError("connection cleanup"), None]
    with pytest.raises(RuntimeError, match="connection cleanup"):
        await agent.close_agent_session(request, close)
    assert opened.agent_session_id in agent._sessions
    await agent.close_agent_session(request, close)
    assert not agent._sessions
    sandbox.stop.assert_not_awaited()


@pytest.mark.asyncio
async def test_cancelled_http_call_does_not_authorize_verification_while_exec_running(sandbox):
    agent = make_agent(session_close_timeout=0.02)
    request = SimpleNamespace(session={}, path_params={"rollout_id": "rollout-a2"})
    started, finished = asyncio.Event(), asyncio.Event()

    async def execute(*args, **kwargs):
        started.set()
        await finished.wait()
        return SandboxExecResult(stdout="late patch", stderr="", return_code=0)

    sandbox.exec.side_effect = execute
    opened = await agent.seed_agent_session(request, seed_request())
    call = asyncio.create_task(agent.responses(request, NeMoGymResponseCreateParamsNonStreaming(input="fix")))
    await started.wait()
    call.cancel()
    with pytest.raises(asyncio.CancelledError):
        await call
    close = AgentCloseSessionRequest(agent_session_id=opened.agent_session_id, episode_id=seed_request().episode_id)
    with pytest.raises(TimeoutError):
        await agent.close_agent_session(request, close)
    assert opened.agent_session_id in agent._sessions
    sandbox.disconnect.assert_not_awaited()
    finished.set()
    await agent.close_agent_session(request, close)
    sandbox.disconnect.assert_awaited_once()
    sandbox.stop.assert_not_awaited()


@pytest.mark.asyncio
async def test_provider_timeout_fails_close_instead_of_assuming_process_stopped(sandbox):
    agent = make_agent()
    request = SimpleNamespace(session={}, path_params={"rollout_id": "rollout-a2"})
    opened = await agent.seed_agent_session(request, seed_request())
    sandbox.exec.return_value = SandboxExecResult(
        stdout="partial", stderr="deadline", return_code=125, error_type="timeout"
    )
    with pytest.raises(RuntimeError, match="timeout"):
        await agent.responses(request, NeMoGymResponseCreateParamsNonStreaming(input="fix"))
    with pytest.raises(RuntimeError, match="unconfirmed"):
        await agent.close_agent_session(
            request,
            AgentCloseSessionRequest(agent_session_id=opened.agent_session_id, episode_id=seed_request().episode_id),
        )
    sandbox.disconnect.assert_not_awaited()
    sandbox.stop.assert_not_awaited()


@pytest.mark.asyncio
async def test_required_tools_rejected_before_connect(monkeypatch, sandbox):
    from nemo_gym.sandboxed_agent import AsyncSandbox

    request = seed_request().model_copy(
        update={
            "tool_accesses": [
                __import__("nemo_gym.tool_access", fromlist=["DirectHTTPToolAccess"]).DirectHTTPToolAccess(
                    name="required", required=True, base_url="http://resources"
                )
            ]
        }
    )
    with pytest.raises(HTTPException) as error:
        await make_agent().seed_agent_session(SimpleNamespace(session={}), request)
    assert error.value.status_code == 422
    AsyncSandbox.connect.assert_not_awaited()
