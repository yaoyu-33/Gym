# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native, single-activation agents borrowing a resources-owned task sandbox."""

import asyncio
import json
import logging
import tempfile
from abc import abstractmethod
from collections import OrderedDict
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar
from uuid import uuid4

from fastapi import Body, FastAPI, HTTPException, Request
from pydantic import Field, JsonValue

from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import (
    AgentCloseSessionRequest,
    AgentCloseSessionResponse,
    AgentSeedSessionRequest,
    AgentSeedSessionResponse,
    BaseResponsesAPIAgentConfig,
    SimpleResponsesAPIAgent,
)
from nemo_gym.config_types import ModelServerRef
from nemo_gym.episode_types import EpisodeId
from nemo_gym.global_config import get_global_config_dict
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.rollout_observability import AgentInvocation, AgentObservationBundle, ObservationGap
from nemo_gym.sandbox import AsyncSandbox, SandboxExecResult
from nemo_gym.sandbox.config import resolve_provider_config
from nemo_gym.sandbox.providers import create_provider


LOG = logging.getLogger(__name__)
_SESSION_KEY = "native_sandboxed_agent_session"


class SandboxedAgentConfig(BaseResponsesAPIAgentConfig):
    """Settings shared by agents that never create or verify task sandboxes."""

    model_server: ModelServerRef
    model: str
    concurrency: int = Field(default=4, ge=1)
    sandbox_timeout: float = Field(default=2700, gt=0)
    session_close_timeout: float = Field(default=30, gt=0)


@dataclass
class SandboxedAgentSession:
    """Retain execution and borrower state until close has completed."""

    seed: AgentSeedSessionRequest
    sandbox: AsyncSandbox
    directory: str
    workdir: str
    activation: asyncio.Task[NeMoGymResponse] | None = None
    executions: set[asyncio.Task[SandboxExecResult]] = field(default_factory=set)
    observations: AgentObservationBundle | None = None
    closing: bool = False
    close_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    execution_uncertain: bool = False


class SandboxedResponsesAPIAgent(SimpleResponsesAPIAgent):
    """Implement Ananth's session protocol without an agent-owned /run workflow.

    Cancellation does not imply a remote process stopped. Keep finite-budget
    exec calls alive and drain them before disconnecting. A close timeout or an
    ambiguous provider result fails closed, so the EnvironmentServer cannot
    verify while a harness may still write. Resources retain stop authority.
    """

    config: SandboxedAgentConfig
    observation_source: ClassVar[str]
    _sessions: dict[str, SandboxedAgentSession]
    _closed_sessions: OrderedDict[str, tuple[EpisodeId, AgentCloseSessionResponse]]
    _sem: asyncio.Semaphore

    def model_post_init(self, context: object) -> None:
        super().model_post_init(context)
        if self.config.num_workers not in (None, 1):
            raise ValueError("Native sandboxed agent sessions require num_workers=1")
        self._sessions = {}
        self._closed_sessions = OrderedDict()
        self._sem = asyncio.Semaphore(self.config.concurrency)

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        parent = app.router.lifespan_context

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            try:
                async with parent(app) as state:
                    yield state
            finally:
                for session_id, session in list(self._sessions.items()):
                    try:
                        await self._close(session)
                    except Exception:
                        LOG.exception("Borrower cleanup failed for %s; resources retain sandbox ownership", session_id)
                    else:
                        del self._sessions[session_id]

        app.router.lifespan_context = lifespan
        return app

    async def seed_agent_session(self, request: Request, body: AgentSeedSessionRequest) -> AgentSeedSessionResponse:
        if request.session.get(_SESSION_KEY) in self._sessions:
            raise HTTPException(409, "An agent session already exists for this cookie")
        if body.sandbox_access is None:
            raise HTTPException(422, "This agent requires resources-owned SandboxAccess; no local fallback")
        required = [access.name for access in self.effective_tool_accesses(body) if access.required]
        if required:
            raise HTTPException(422, f"Required runtime HTTP/MCP tools are not supported: {required}")
        connection = body.sandbox_access.connection
        provider = create_provider(resolve_provider_config(connection.provider_config_ref, get_global_config_dict()))
        try:
            sandbox = await AsyncSandbox.connect(connection.descriptor, provider=provider)
        except BaseException:
            await provider.aclose()
            raise
        session_id = f"sandbox-agent-{uuid4().hex}"
        session = SandboxedAgentSession(body, sandbox, f"/tmp/{session_id}", body.sandbox_access.workdir)
        # Keep failed setup reachable for shutdown if disconnect itself fails.
        self._sessions[session_id] = session
        try:
            await self.prepare_session(session)
        except BaseException:
            try:
                await self._close(session)
            except Exception:
                LOG.exception("Failed to release sandbox borrower after setup failure")
            else:
                del self._sessions[session_id]
            raise
        request.session[_SESSION_KEY] = session_id
        return AgentSeedSessionResponse(agent_session_id=session_id)

    @abstractmethod
    async def prepare_session(self, session: SandboxedAgentSession) -> None:
        """Check the pinned runtime and prepare agent-owned files, not the task."""

    @abstractmethod
    async def execute_response(
        self, session: SandboxedAgentSession, request: Request, body: NeMoGymResponseCreateParamsNonStreaming
    ) -> NeMoGymResponse:
        """Run the harness in session.workdir and return its actual trajectory."""

    async def responses(
        self, request: Request, body: NeMoGymResponseCreateParamsNonStreaming = Body()
    ) -> NeMoGymResponse:
        session_id = request.session.get(_SESSION_KEY)
        session = self._sessions.get(session_id)
        if session is None:
            raise HTTPException(409, "Seed an agent session through the EnvironmentServer first")
        if request.path_params.get("rollout_id") != session.seed.episode_id.capture_key:
            raise HTTPException(409, "Rollout route does not match the seeded episode")
        if session.activation is not None or session.closing:
            raise HTTPException(409, "This session is already activated or closing")

        async def activate() -> NeMoGymResponse:
            async with self._sem:
                if session.closing:
                    raise HTTPException(409, "Agent session is closing")
                response = await self.execute_response(session, request, body)
                if session.observations is None:
                    session.observations = AgentObservationBundle(
                        source=self.observation_source,
                        records=[
                            AgentInvocation(
                                invocation_id=session.seed.episode_id.capture_key, conversation=response.output
                            )
                        ],
                        gaps=[
                            ObservationGap(code="model_call_ownership_unavailable"),
                            ObservationGap(code="tool_timing_unavailable"),
                        ],
                    )
                return response

        session.activation = asyncio.create_task(activate())
        return await asyncio.shield(session.activation)

    async def close_agent_session(self, request: Request, body: AgentCloseSessionRequest) -> AgentCloseSessionResponse:
        if body.agent_session_id != request.session.get(_SESSION_KEY):
            raise HTTPException(409, "Agent session does not match its cookie")
        closed = self._closed_sessions.get(body.agent_session_id)
        if closed is not None:
            if closed[0] != body.episode_id:
                raise HTTPException(409, "Close episode does not match its original session")
            return closed[1]
        session = self._sessions.get(body.agent_session_id)
        if session is None or body.episode_id != session.seed.episode_id:
            raise HTTPException(409, "Unknown agent session or mismatched episode")
        await self._close(session)
        self._sessions.pop(body.agent_session_id, None)
        response = AgentCloseSessionResponse(
            agent_session_id=body.agent_session_id, agent_observations=session.observations
        )
        # Retain the cookie binding and a bounded acknowledgement for HTTP
        # retries after a successful close whose response was lost.
        self._closed_sessions[body.agent_session_id] = (body.episode_id, response)
        if len(self._closed_sessions) > 1024:
            self._closed_sessions.popitem(last=False)
        return response

    async def _close(self, session: SandboxedAgentSession) -> None:
        async with session.close_lock:
            session.closing = True
            async with asyncio.timeout(self.config.session_close_timeout):
                if session.activation is not None:
                    try:
                        await asyncio.shield(session.activation)
                    except Exception:
                        # /responses already reports the activation error. Outstanding
                        # execs and uncertain termination are checked independently.
                        pass
                for task in list(session.executions):
                    result = await asyncio.shield(task)
                    if result.error_type:
                        session.execution_uncertain = True
                    session.executions.discard(task)
                if session.execution_uncertain:
                    raise RuntimeError("Remote command termination is unconfirmed; verification must not proceed")
                await session.sandbox.disconnect()

    async def exec_in_session(
        self,
        session: SandboxedAgentSession,
        command: str,
        *,
        cwd: str | None = None,
        timeout_s: float = 30,
        env: dict[str, str] | None = None,
    ) -> SandboxExecResult:
        """Do not lose a remote command when its HTTP caller is cancelled."""
        task = asyncio.create_task(
            session.sandbox.exec(command, cwd=cwd or session.workdir, env=env, timeout_s=timeout_s)
        )
        session.executions.add(task)
        try:
            result = await asyncio.shield(task)
        except asyncio.CancelledError:
            raise
        except Exception:
            session.execution_uncertain = True
            raise
        else:
            session.executions.discard(task)
            if result.error_type:
                session.execution_uncertain = True
                raise RuntimeError(f"Sandbox execution {result.error_type}: {result.stderr or ''}")
            return result

    @staticmethod
    async def upload_text(session: SandboxedAgentSession, name: str, text: str) -> None:
        """Upload an agent artifact under the random session directory."""
        with tempfile.TemporaryDirectory(prefix="gym-agent-upload-") as directory:
            local = Path(directory) / "artifact"
            local.write_text(text, encoding="utf-8")
            await session.sandbox.upload(local, f"{session.directory}/{name}")

    @staticmethod
    async def download_text(session: SandboxedAgentSession, name: str) -> str:
        with tempfile.TemporaryDirectory(prefix="gym-agent-download-") as directory:
            local = Path(directory) / "artifact"
            await session.sandbox.download(f"{session.directory}/{name}", local)
            return local.read_text(encoding="utf-8", errors="replace")

    async def download_json(self, session: SandboxedAgentSession, name: str) -> dict[str, JsonValue]:
        value = json.loads(await self.download_text(session, name))
        if not isinstance(value, dict):
            raise ValueError(f"Expected a JSON object in {name}")
        return value

    async def run(self, request: Request, body: BaseRunRequest = Body()) -> BaseVerifyResponse:
        raise HTTPException(409, "Use the EnvironmentServer /run endpoint; this agent only implements sessions")
