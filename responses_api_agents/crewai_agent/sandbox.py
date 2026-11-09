# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Agent-server side of a CrewAI crew run in a task sandbox."""

import asyncio
import importlib.metadata
import json
from dataclasses import dataclass, field
from pathlib import Path
from shlex import quote
from typing import Any

from nemo_gym.agents import AgentSessionState, SandboxCommand, SandboxPythonRuntime, SandboxSession
from nemo_gym.core import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming, NgFailureEvent
from nemo_gym.sandbox import read_text, upload_text


# The sandbox runs the CrewAI version installed with this server, with its dependencies at this server's versions.
RUNTIME = SandboxPythonRuntime(
    harness="crewai",
    requirements=tuple(
        f"{package}=={importlib.metadata.version(package)}" for package in ("crewai", "aiohttp", "mcp")
    ),
    import_check="import crewai, aiohttp, mcp",
)
# Outside the working directory, so the model's files stay its own.
SANDBOX_PYTHON = RUNTIME.python
_PACKAGE_DIR = f"{RUNTIME.directory}/harness/responses_api_agents/crewai_agent"
# The runner and the modules it imports; the rest of this package needs Gym, which the sandbox lacks.
HARNESS_FILES = ("crew.py", "mcp_client.py", "sandbox_runner.py")
SESSION_ROOT = "/tmp/nemo-gym-crewai-sessions"


@dataclass
class CrewAISessionState(AgentSessionState):
    """One agent session whose crew runs in a task sandbox, and its single activation."""

    session: SandboxSession[dict[str, Any]]
    # The episode's MCP grants as the runner reaches them: name, url, headers and whether the episode requires it.
    grants: list[dict[str, Any]]
    # The sandbox's provider and ID, kept for the close bundle: a released sandbox no longer reports them.
    sandbox_provider: str | None = None
    sandbox_id: str | None = None
    provider_wait_failure: NgFailureEvent | None = None
    infrastructure_failure: NgFailureEvent | None = None
    activation_request: NeMoGymResponseCreateParamsNonStreaming | None = None
    task: asyncio.Task[NeMoGymResponse] | None = field(default=None)

    async def install_runtime(self, *, timeout: float) -> None:
        """Reuse or install the CrewAI runtime and stage the harness files."""
        sandbox = self.session.sandbox
        prepared = await sandbox.exec(
            f"mkdir -p {quote(_PACKAGE_DIR)} {quote(self.session.session_dir)}", cwd="/", timeout_s=30
        )
        if prepared.return_code != 0 or prepared.error_type is not None:
            raise RuntimeError(prepared.stderr or prepared.stdout or "Failed to prepare CrewAI sandbox paths")
        if not await RUNTIME.is_installed(sandbox):
            await RUNTIME.install(sandbox, timeout_s=timeout)
        for name in HARNESS_FILES:
            await sandbox.upload(Path(__file__).with_name(name), f"{_PACKAGE_DIR}/{name}")
        for package in (_PACKAGE_DIR, str(Path(_PACKAGE_DIR).parent)):
            await upload_text(sandbox, path=f"{package}/__init__.py", text="")

    async def execute(self, payload: dict[str, Any], *, timeout: float, close_timeout: float) -> dict[str, Any]:
        """Run the crew once and return the runner's report."""
        input_path = f"{self.session.session_dir}/input.json"
        output_path = f"{self.session.session_dir}/output.json"

        async def stage() -> SandboxCommand:
            await upload_text(self.session.sandbox, path=input_path, text=json.dumps(payload))
            # Ignore task Python settings without changing the environment inherited by tool subprocesses.
            return SandboxCommand(
                argv=[SANDBOX_PYTHON, "-E", "-s", f"{_PACKAGE_DIR}/sandbox_runner.py", input_path, output_path]
            )

        return await self.session.execute(
            stage_activation=stage,
            collect=lambda: self.collect_report(output_path),
            timeout=timeout,
            close_timeout=close_timeout,
        )

    async def collect_report(self, path: str) -> dict[str, Any]:
        """The runner's report at ``path``. A runner that left none (killed, or never started; it writes one on every
        exit it can catch) gets a report of its own failure: the crew left no usable result. A report that is there
        but cannot be read is a lifecycle failure."""
        try:
            return json.loads(await read_text(self.session.sandbox, path=path))
        except Exception as error:
            logs = await self.session.read_output_log()
            if await self._report_absent(path):
                return {"status": "failed", "error": "exited without a report", "traceback": logs[-4000:]}
            raise RuntimeError(f"CrewAI sandbox runner exited without a report: {logs}") from error

    async def _report_absent(self, path: str) -> bool:
        """Whether there is no file at ``path``, as opposed to a read that failed."""
        try:
            probe = await self.session.sandbox.exec(f"[ -e {quote(path)} ] || echo absent", cwd="/", timeout_s=10)
        except Exception:
            return False
        return probe.return_code == 0 and probe.error_type is None and (probe.stdout or "").strip() == "absent"

    async def close(self, timeout: float) -> None:
        """Stop the crew and release the sandbox before cancelling the HTTP activation."""
        await self.session.close(timeout=timeout)
        if self.task is not None and not self.task.done():
            self.task.cancel()
            try:
                await asyncio.wait_for(asyncio.shield(self.task), timeout=timeout)
            except (asyncio.CancelledError, Exception):
                if not self.task.done():
                    raise
