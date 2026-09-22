# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from fastapi import Request

from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.sandboxed_agent import SandboxedAgentSession
from nemo_gym.sandboxed_cli_agent import SandboxedCLIAgent, SandboxedCLIConfig, text_prompt
from nemo_gym.server_utils import is_nemo_gym_fastapi_entrypoint
from responses_api_agents.codex_agent.app import parse_exec_jsonl, toml_dumps


class CodexSandboxedAgentConfig(SandboxedCLIConfig):
    cli_version: str = "0.144.4"
    reasoning_effort: str | None = None


class CodexSandboxedAgent(SandboxedCLIAgent):
    config: CodexSandboxedAgentConfig
    observation_source = "codex"
    cli_name = "codex"

    async def execute_response(
        self, session: SandboxedAgentSession, request: Request, body: NeMoGymResponseCreateParamsNonStreaming
    ) -> NeMoGymResponse:
        prompt, system = text_prompt(body, self.config.system_prompt)
        if body.temperature is not None or body.max_output_tokens is not None:
            raise ValueError(
                "Pinned Codex does not expose these request sampling overrides; configure the Gym model server"
            )
        config = {
            "model_provider": "gym",
            "model": self.config.model,
            "approval_policy": "never",
            "sandbox_mode": "danger-full-access",
            "web_search": "disabled",
            "check_for_update_on_startup": False,
            "analytics": {"enabled": False},
            "history": {"persistence": "none"},
            "features": {"multi_agent": False, "code_mode": False},
            "model_providers": {
                "gym": {
                    "name": "gym",
                    "base_url": self.resolve_model_base_url(
                        self.config.model_server.name, session.seed.episode_id.capture_key
                    ),
                    "env_key": "OPENAI_API_KEY",
                    "wire_api": "responses",
                    "stream_idle_timeout_ms": int(self.config.sandbox_timeout * 1000),
                }
            },
        }
        if system:
            config["developer_instructions"] = system
        if self.config.reasoning_effort:
            config["model_reasoning_effort"] = self.config.reasoning_effort
        await self.upload_text(session, "home/.codex/config.toml", toml_dumps(config))
        command = [
            self.cli,
            "exec",
            "--json",
            "--ephemeral",
            "--skip-git-repo-check",
            "--cd",
            session.workdir,
            "--",
            prompt,
        ]
        summary, stdout = await self.run_cli(
            session, command, env={"CODEX_HOME": session.directory + "/home/.codex", "OPENAI_API_KEY": "gym"}
        )
        output, usage = parse_exec_jsonl(stdout)
        return self.response_from_output(
            body, [item.model_dump(mode="json") for item in output], usage, summary, session=session
        )


if __name__ == "__main__":
    CodexSandboxedAgent.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = CodexSandboxedAgent.run_webserver()
