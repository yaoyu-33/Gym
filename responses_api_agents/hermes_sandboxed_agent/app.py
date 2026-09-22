# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the portable Hermes runtime through native EnvironmentServer sessions."""

import json
from pathlib import Path
from shlex import quote

from fastapi import Request
from pydantic import Field, JsonValue

from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.sandboxed_agent import SandboxedAgentConfig, SandboxedAgentSession, SandboxedResponsesAPIAgent
from nemo_gym.server_utils import is_nemo_gym_fastapi_entrypoint
from responses_api_agents.hermes_sandboxed_agent.runner import classify_stop, split_input
from responses_api_agents.hermes_sandboxed_agent.trajectory import trajectory_response


class HermesSandboxedAgentConfig(SandboxedAgentConfig):
    """Pinned Hermes lives in the task image or a resources-configured read-only mount."""

    runtime_python: str = Field(default="/opt/hermes/bin/hermes-python", pattern=r"^/")
    hermes_commit: str = Field(default="2237be355906fbe6065ce1815711eee52b2d646e", pattern=r"^[0-9a-f]{40}$")
    api_timeout: float = Field(default=1800, gt=0)
    max_turns: int = Field(default=90, gt=0)
    max_tokens: int | None = Field(default=None, gt=0)
    context_length: int | None = Field(default=None, gt=0)
    temperature: float = 1.0
    terminal_timeout: int = Field(default=180, gt=0)
    enabled_toolsets: list[str] | None = None
    disabled_toolsets: list[str] | None = None
    compression_enabled: bool = True
    system_prompt: str | None = None
    chat_template_kwargs: dict[str, JsonValue] = Field(default_factory=dict)


class HermesSandboxedAgent(SandboxedResponsesAPIAgent):
    config: HermesSandboxedAgentConfig
    observation_source = "hermes"

    async def prepare_session(self, session: SandboxedAgentSession) -> None:
        # A remote task runtime is unavailable at server startup. Check it at
        # seed, before admitting any activation, using the actual container ABI.
        probe = (
            "import inspect,json,pathlib,sys; import run_agent; "
            "root=pathlib.Path(sys.prefix); "
            "manifest=json.loads((root/'hermes-runtime.json').read_text()); "
            "assert manifest['hermes_commit']==sys.argv[1], 'Hermes version mismatch'; "
            "assert pathlib.Path(run_agent.__file__).resolve().is_relative_to((root/'hermes-src').resolve()); "
            "assert 'request_overrides' in inspect.signature(run_agent.AIAgent).parameters; "
            "assert (root/'tools/bin/rg').is_file()"
        )
        checked = await self.exec_in_session(
            session,
            f"{quote(self.config.runtime_python)} -I -c {quote(probe)} {quote(self.config.hermes_commit)}",
            cwd="/tmp",
        )
        if checked.return_code:
            raise RuntimeError(f"Prepare the pinned Hermes runtime before seeding: {checked.stderr or checked.stdout}")
        created = await self.exec_in_session(session, f"mkdir -p {quote(session.directory)}", cwd="/tmp")
        if created.return_code:
            raise RuntimeError("Cannot prepare Hermes session directory")
        await session.sandbox.upload(Path(__file__).with_name("runner.py"), f"{session.directory}/runner.py")

    def request_parameters(
        self, session: SandboxedAgentSession, request: Request, body: NeMoGymResponseCreateParamsNonStreaming
    ) -> dict[str, JsonValue]:
        """Explicit request sampling parameters override config defaults, including zero."""
        serialized = body.model_dump(mode="json")
        split_input(serialized["input"])
        params = self.config.model_dump(
            include={
                "model",
                "max_turns",
                "max_tokens",
                "context_length",
                "temperature",
                "terminal_timeout",
                "api_timeout",
                "enabled_toolsets",
                "disabled_toolsets",
                "compression_enabled",
                "system_prompt",
                "chat_template_kwargs",
            },
            mode="json",
        )
        if body.temperature is not None:
            params["temperature"] = body.temperature
        if body.max_output_tokens is not None:
            params["max_tokens"] = body.max_output_tokens
        metadata = body.metadata or {}
        if metadata.get("chat_template_kwargs"):
            overrides = json.loads(metadata["chat_template_kwargs"])
            if not isinstance(overrides, dict):
                raise ValueError("metadata.chat_template_kwargs must encode an object")
            params["chat_template_kwargs"] = self.config.chat_template_kwargs | overrides
        params.update(
            wall_time=self.config.sandbox_timeout,
            run_dir=session.directory,
            workdir=session.workdir,
            base_url=self.resolve_model_base_url(self.config.model_server.name, session.seed.episode_id.capture_key),
            input=serialized["input"],
        )
        return params

    async def execute_response(
        self, session: SandboxedAgentSession, request: Request, body: NeMoGymResponseCreateParamsNonStreaming
    ) -> NeMoGymResponse:
        params = self.request_parameters(session, request, body)
        await self.upload_text(session, "request.json", json.dumps(params))
        executed = await self.exec_in_session(
            session,
            f"{quote(self.config.runtime_python)} -I {quote(session.directory + '/runner.py')} "
            f"{quote(session.directory + '/request.json')}",
            cwd=session.directory,
            timeout_s=self.config.sandbox_timeout + self.config.session_close_timeout + 30,
        )
        result = await self.download_json(session, "result.json")
        result = classify_stop(result)
        if result.get("cleanup_confirmed") is False:
            session.execution_uncertain = True
        response = trajectory_response(
            result, body, self.config.model, "runner_exit" if executed.return_code else None
        )
        response.metadata = (response.metadata or {}) | {
            "harness_execution": "sandbox",
            "hermes_commit": self.config.hermes_commit,
        }
        # Resources, not the harness adapter, decide reward and masking. Failed
        # responses with a usable patch still travel through close -> verify.
        return response


if __name__ == "__main__":
    HermesSandboxedAgent.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = HermesSandboxedAgent.run_webserver()
