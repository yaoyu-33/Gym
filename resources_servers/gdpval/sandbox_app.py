# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prototype Resources-owned GDP sandbox lifecycle; reuse the existing judge."""

import asyncio
import json
import logging
import re
import tempfile
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from shlex import quote

from aiohttp import ClientTimeout
from fastapi import FastAPI, HTTPException, Request
from pydantic import Field, PrivateAttr, field_validator

from nemo_gym.base_resources_server import (
    ResourcesCloseSessionRequest,
    ResourcesCloseSessionResponse,
    ResourcesSeedSessionRequest,
    ResourcesSeedSessionResponse,
)
from nemo_gym.episode_types import EpisodeId
from nemo_gym.global_config import get_global_config_dict
from nemo_gym.sandbox import AsyncSandbox, SandboxSpec, resolve_provider_config
from nemo_gym.sandbox.access import DirectSandboxConnection, SandboxAccess
from nemo_gym.server_utils import SESSION_ID_KEY, is_nemo_gym_fastapi_entrypoint
from nemo_gym.server_utils import request as http_request
from resources_servers.gdpval.app import (
    GDPValResourcesServer,
    GDPValResourcesServerConfig,
    GDPValVerifyRequest,
    GDPValVerifyResponse,
)
from resources_servers.gdpval.sandbox_tasks import INPUT_DIR, OUTPUT_DIR, WORKDIR, GDPFileTask, relative_file


LOG = logging.getLogger(__name__)
_MAX_BYTES = 128 * 1024 * 1024
_LIST_OUTPUTS = f"""
import json, pathlib, stat
root = pathlib.Path({OUTPUT_DIR!r})
if root.is_symlink() or not root.is_dir():
    raise RuntimeError('Output directory is missing or a symlink')
files = []
for path in sorted(root.iterdir()):
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise RuntimeError('Only regular, non-linked files directly in output are supported')
    files.append({{'name': path.name, 'size': info.st_size}})
if len(files) > 100 or sum(item['size'] for item in files) > {_MAX_BYTES}:
    raise RuntimeError('Deliverable limit exceeded')
print(json.dumps(files))
"""


class GDPSandboxConfig(GDPValResourcesServerConfig):
    """Use one worker and an explicitly provisioned GDP image."""

    sandbox_provider: str = "sandbox"
    image: str = Field(min_length=1)
    deliverables_root: Path
    num_workers: int = 1

    @field_validator("deliverables_root")
    @classmethod
    def absolute_output(cls, value: Path) -> Path:
        if not value.is_absolute():
            raise ValueError("deliverables_root must be absolute")
        return value

    @field_validator("num_workers")
    @classmethod
    def one_worker(cls, value: int) -> int:
        if value != 1:
            raise ValueError("The prototype uses process-local sessions; num_workers must be 1")
        return value


@dataclass
class _Session:
    seed: ResourcesSeedSessionRequest
    sandbox: AsyncSandbox
    ready: bool = False
    deliverables: Path | None = None
    verdict: GDPValVerifyResponse | None = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class GDPSandboxResourcesServer(GDPValResourcesServer):
    """Prepare a task, lend its sandbox, export files after agent close, then grade."""

    config: GDPSandboxConfig
    _sessions: dict[str, _Session] = PrivateAttr(default_factory=dict)
    _closed: dict[str, EpisodeId] = PrivateAttr(default_factory=dict)

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        parent = app.router.lifespan_context

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            try:
                async with parent(app) as state:
                    yield state
            finally:
                for session in list(self._sessions.values()):
                    try:
                        async with asyncio.timeout(60):
                            await session.sandbox.stop()
                    except Exception:
                        LOG.exception("Failed to stop GDP sandbox during shutdown")

        app.router.lifespan_context = lifespan
        return app

    async def seed_session(self, request: Request, body: ResourcesSeedSessionRequest) -> ResourcesSeedSessionResponse:
        session_id = body.resources_session_id
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", session_id):
            raise HTTPException(422, "Invalid resources_session_id")
        task = GDPFileTask.model_validate(body.task_data)
        if task.task_id != body.task_id.task_id:
            raise HTTPException(422, "Task ID does not match task_data")
        if session_id in self._closed:
            raise HTTPException(409, "Resources session is already closed")
        session = self._sessions.get(session_id)
        if session is None:
            provider = resolve_provider_config(self.config.sandbox_provider, get_global_config_dict())
            session = _Session(body.model_copy(deep=True), AsyncSandbox(provider))
            self._sessions[session_id] = session
        if session.seed != body:
            raise HTTPException(409, "Session is already bound to another request")
        async with session.lock:
            if session_id in self._closed:
                raise HTTPException(409, "Resources session is already closed")
            if not session.ready:
                try:
                    await session.sandbox.start(SandboxSpec(image=self.config.image, workdir=WORKDIR))
                    result = await session.sandbox.exec(f"mkdir -p {INPUT_DIR} {OUTPUT_DIR}", timeout_s=30)
                    if result.return_code != 0:
                        raise RuntimeError("Could not prepare GDP sandbox directories")
                    await self._stage_references(session.sandbox, task)
                    session.ready = True
                except BaseException:
                    # Leave the handle reachable if stop fails; close_session can retry.
                    try:
                        await session.sandbox.stop()
                    except BaseException:
                        LOG.exception("GDP seed cleanup failed; retaining session %s", session_id)
                    raise
            descriptor = await session.sandbox.serialize()
            request.session[SESSION_ID_KEY] = session_id
            return ResourcesSeedSessionResponse(
                resources_session_id=session_id,
                sandbox_access=SandboxAccess(
                    connection=DirectSandboxConnection(
                        provider_config_ref=self.config.sandbox_provider, descriptor=descriptor
                    ),
                    workdir=WORKDIR,
                ),
            )

    async def _stage_references(self, sandbox: AsyncSandbox, task: GDPFileTask) -> None:
        with tempfile.TemporaryDirectory(prefix="gdp-input-") as scratch:
            for name, url in zip(task.reference_files, task.reference_file_urls, strict=True):
                response = await http_request("GET", url, timeout=ClientTimeout(total=180))
                try:
                    response.raise_for_status()
                    local = Path(scratch) / "reference"
                    size = 0
                    with local.open("wb") as stream:
                        async for chunk in response.content.iter_chunked(1024 * 1024):
                            size += len(chunk)
                            if size > _MAX_BYTES:
                                raise RuntimeError("Reference file exceeds prototype download limit")
                            stream.write(chunk)
                    await sandbox.upload(local, f"{INPUT_DIR}/{name}")
                finally:
                    response.release()

    async def export_deliverables(self, session_id: str) -> Path:
        """Export completed files; caller must have confirmed agent close first."""
        session = self._sessions.get(session_id)
        if session is None or not session.ready:
            raise HTTPException(409, "No ready GDP sandbox for this session")
        if session.deliverables is not None:
            return session.deliverables
        result = await session.sandbox.exec(f"python3 -c {quote(_LIST_OUTPUTS)}", timeout_s=60)
        if result.return_code != 0:
            raise HTTPException(503, "GDP artifact export failed: " + (result.stderr or "listing failed")[-1000:])
        files = json.loads(result.stdout)
        if not isinstance(files, list) or len(files) > 100:
            raise HTTPException(503, "Invalid GDP artifact listing")
        self.config.deliverables_root.mkdir(parents=True, exist_ok=True)
        # An attempt gets a fresh directory. Never delete or overwrite another attempt's files.
        target = Path(tempfile.mkdtemp(prefix="gdp-", dir=self.config.deliverables_root))
        total = 0
        for item in files:
            name = relative_file(item["name"])
            if "/" in name or not isinstance(item["size"], int) or item["size"] < 0:
                raise HTTPException(503, "Invalid GDP artifact entry")
            total += item["size"]
            if total > _MAX_BYTES:
                raise HTTPException(503, "GDP artifact size limit exceeded")
            await session.sandbox.download(f"{OUTPUT_DIR}/{name}", target / name)
            if (target / name).stat().st_size != item["size"]:
                raise HTTPException(503, "GDP artifact changed during export")
        session.deliverables = target
        return target

    async def verify(self, request: Request, body: GDPValVerifyRequest) -> GDPValVerifyResponse:
        session_id = request.session.get(SESSION_ID_KEY)
        session = self._sessions.get(session_id)
        if session is None or not session.ready or body.task_id != session.seed.task_id.task_id:
            raise HTTPException(409, "Verification does not match a ready GDP session")
        async with session.lock:
            if session.verdict is not None:
                return session.verdict
            target = await self.export_deliverables(session_id)
            # Trust seeded task metadata, never a caller-supplied rubric or host directory.
            payload = GDPValVerifyRequest.model_validate(
                session.seed.task_data
                | {
                    "responses_create_params": body.responses_create_params,
                    "response": body.response,
                    "deliverables_dir": str(target),
                }
            )
            verdict = await super().verify(payload)
            if verdict.invalid_judge_response:
                raise HTTPException(503, "GDP judge did not return a valid verdict")
            session.verdict = verdict
            return session.verdict

    async def close_resources_session(
        self, request: Request, body: ResourcesCloseSessionRequest
    ) -> ResourcesCloseSessionResponse:
        session_id = body.resources_session_id
        closed = self._closed.get(session_id)
        if closed is not None and closed != body.episode_id:
            raise HTTPException(409, "Close episode does not match")
        session = self._sessions.get(session_id)
        if session is not None:
            if session.seed.episode_id != body.episode_id:
                raise HTTPException(409, "Close episode does not match")
            async with session.lock:
                async with asyncio.timeout(60):
                    await session.sandbox.stop()
                self._sessions.pop(session_id, None)
        self._closed[session_id] = body.episode_id
        request.session.pop(SESSION_ID_KEY, None)
        return ResourcesCloseSessionResponse(resources_session_id=session_id)


if __name__ == "__main__":
    GDPSandboxResourcesServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = GDPSandboxResourcesServer.run_webserver()
