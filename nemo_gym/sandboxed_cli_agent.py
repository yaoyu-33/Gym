# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Small execution adapter shared by native sandboxed CLI harnesses."""

import json
import re
from pathlib import Path
from shlex import join, quote
from time import time
from typing import ClassVar
from uuid import uuid4

from pydantic import Field, JsonValue

from nemo_gym.openai_utils import NeMoGymEasyInputMessage, NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.rollout_observability import AgentInvocation, AgentObservationBundle, ObservationGap
from nemo_gym.sandboxed_agent import SandboxedAgentConfig, SandboxedAgentSession, SandboxedResponsesAPIAgent


class SandboxedCLIConfig(SandboxedAgentConfig):
    """A prepared, ABI-compatible runtime; no host CLI or runtime installation."""

    runtime_python: str = Field(default="/opt/gym-cli/bin/python3", pattern=r"^/")
    runtime_bin: str = Field(default="/opt/gym-cli/launchers", pattern=r"^/")
    cli_version: str = Field(pattern=r"^[0-9]+\.[0-9]+\.[0-9]+(?:[-+][A-Za-z0-9.-]+)?$")
    system_prompt: str | None = None


def text_prompt(body: NeMoGymResponseCreateParamsNonStreaming, system_prompt: str | None) -> tuple[str, str | None]:
    """Support text-only SWE tasks and reject unsupported history rather than dropping it."""
    serialized = body.model_dump(mode="json")
    # The CLI owns its tools and generation loop. Never silently reinterpret
    # request-level controls that its pinned adapter cannot honor.
    for name in (
        "top_p",
        "reasoning",
        "max_tool_calls",
        "previous_response_id",
        "prompt",
        "text",
        "context_management",
        "conversation",
        "moderation",
        "top_logprobs",
        "truncation",
    ):
        if serialized.get(name) is not None:
            raise ValueError(f"Sandboxed CLI adapter does not support request field {name}")
    if body.tools or body.tool_choice != "auto" or not body.parallel_tool_calls or body.background:
        raise ValueError("Sandboxed CLI tool policy is controlled by the harness, not the Responses request")
    if body.metadata and "chat_template_kwargs" in body.metadata:
        raise ValueError("Set chat_template_kwargs on the Gym model server for sandboxed CLI agents")
    payload = serialized["input"]
    system = [text for text in (system_prompt, body.instructions) if text]
    if isinstance(payload, str):
        return payload, "\n\n".join(system) or None
    users = []
    for item in payload:
        if item.get("role") not in ("system", "developer", "user"):
            raise ValueError("Sandboxed CLI inputs currently require a single user turn and optional system text")
        content = item.get("content", "")
        if isinstance(content, list):
            if any(part.get("type") not in ("input_text", "output_text", "text") for part in content):
                raise ValueError("Sandboxed CLI inputs support text only")
            content = "\n".join(part["text"] for part in content)
        (users if item["role"] == "user" else system).append(content)
    if len(users) != 1:
        raise ValueError("Sandboxed CLI inputs require exactly one user message")
    return users[0], "\n\n".join(system) or None


class SandboxedCLIAgent(SandboxedResponsesAPIAgent):
    config: SandboxedCLIConfig
    cli_name: ClassVar[str]

    @property
    def cli(self) -> str:
        return f"{self.config.runtime_bin}/{self.cli_name}"

    async def prepare_session(self, session: SandboxedAgentSession) -> None:
        checked = await self.exec_in_session(session, f"{quote(self.cli)} --version", cwd="/tmp")
        version = (checked.stdout or "") + (checked.stderr or "")
        if checked.return_code or not re.search(
            r"(?<![\d.])" + re.escape(self.config.cli_version) + r"(?![\d.])", version
        ):
            raise RuntimeError(
                f"Prepare {self.cli_name} {self.config.cli_version} in the task runtime; got {version!r}"
            )
        checked = await self.exec_in_session(
            session, f"{quote(self.config.runtime_python)} -I -c 'import subprocess,json'", cwd="/tmp"
        )
        if checked.return_code:
            raise RuntimeError("The sandboxed CLI runner needs its prepared Python runtime")
        created = await self.exec_in_session(session, f"mkdir -p {quote(session.directory + '/home')}", cwd="/tmp")
        if created.return_code:
            raise RuntimeError("Cannot create CLI session directory")
        await session.sandbox.upload(
            Path(__file__).with_name("sandboxed_cli_runner.py"), f"{session.directory}/runner.py"
        )

    async def run_cli(
        self,
        session: SandboxedAgentSession,
        command: list[str],
        *,
        env: dict[str, str],
        timeout_s: float | None = None,
    ) -> tuple[dict[str, JsonValue], str]:
        """Run with a task-local HOME and a bounded process-group cleanup grace."""
        timeout = self.config.sandbox_timeout if timeout_s is None else timeout_s
        payload = {
            "command": command,
            "cwd": session.workdir,
            "directory": session.directory,
            "env": {"HOME": session.directory + "/home", **env},
            "timeout": timeout,
            "cleanup_timeout": self.config.session_close_timeout,
        }
        await self.upload_text(session, "command.json", json.dumps(payload))
        result = await self.exec_in_session(
            session,
            join(
                [
                    self.config.runtime_python,
                    "-I",
                    session.directory + "/runner.py",
                    session.directory + "/command.json",
                ]
            ),
            cwd=session.directory,
            timeout_s=timeout + self.config.session_close_timeout + 30,
        )
        summary = await self.download_json(session, "result.json")
        if result.return_code or summary.get("cleanup_confirmed") is not True:
            session.execution_uncertain = True
            raise RuntimeError(f"{self.cli_name} runner could not confirm cleanup")
        if summary.get("error"):
            raise RuntimeError(f"{self.cli_name} runner failed before producing a usable result: {summary['error']}")
        stdout = await self.download_text(session, "stdout.log")
        return summary, stdout

    def response_from_output(
        self,
        body: NeMoGymResponseCreateParamsNonStreaming,
        output: list[dict[str, JsonValue]],
        usage: dict[str, JsonValue],
        summary: dict[str, JsonValue],
        *,
        session: SandboxedAgentSession,
    ) -> NeMoGymResponse:
        """Keep real output/usage; no fabricated assistant success or local scoring."""
        timed_out = summary.get("timed_out") is True
        budget_exhausted = timed_out or summary.get("budget_exhausted") is True
        failed = (bool(summary.get("return_code")) and not timed_out) or not output or bool(usage.get("errors"))
        inputs, outputs = int(usage.get("input_tokens") or 0), int(usage.get("output_tokens") or 0)
        response = NeMoGymResponse.model_validate(
            {
                "id": f"resp_{uuid4().hex}",
                "created_at": int(time()),
                "object": "response",
                "model": self.config.model,
                "status": "failed" if failed else "incomplete" if budget_exhausted else "completed",
                "output": output,
                "error": {"code": "server_error", "message": f"{self.cli_name} exited without a complete result"}
                if failed
                else None,
                "metadata": {
                    "harness_execution": "sandbox",
                    "harness_version": self.config.cli_version,
                    "budget_exhausted": str(budget_exhausted and not failed).lower(),
                },
                "tools": body.tools,
                "tool_choice": body.tool_choice,
                "parallel_tool_calls": body.parallel_tool_calls,
                "usage": {
                    "input_tokens": inputs,
                    "output_tokens": outputs,
                    "total_tokens": inputs + outputs,
                    "input_tokens_details": {
                        "cached_tokens": int(usage.get("cached_tokens") or usage.get("cached_input_tokens") or 0)
                    },
                    "output_tokens_details": {"reasoning_tokens": int(usage.get("reasoning_tokens") or 0)},
                },
            }
        )
        inputs = (
            [NeMoGymEasyInputMessage(role="user", content=body.input)]
            if isinstance(body.input, str)
            else list(body.input)
        )
        session.observations = AgentObservationBundle(
            source=self.observation_source,
            records=[
                AgentInvocation(
                    invocation_id=session.seed.episode_id.capture_key,
                    status=response.status,
                    conversation=[*inputs, *response.output],
                )
            ],
            gaps=[
                ObservationGap(code=code)
                for code in (
                    "model_call_ownership_unavailable",
                    "tool_timing_unavailable",
                    "subagent_hierarchy_unavailable",
                )
            ],
        )
        return response
