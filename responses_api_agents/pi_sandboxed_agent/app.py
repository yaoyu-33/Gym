# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

from fastapi import Request
from pydantic import Field

from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseReasoningItem,
)
from nemo_gym.sandboxed_agent import SandboxedAgentSession
from nemo_gym.sandboxed_cli_agent import SandboxedCLIAgent, SandboxedCLIConfig, text_prompt
from nemo_gym.server_utils import is_nemo_gym_fastapi_entrypoint
from responses_api_agents.pi_agent.app import parse_pi_events


class PiSandboxedAgentConfig(SandboxedCLIConfig):
    cli_version: str = "0.80.2"
    context_window: int = Field(default=262144, gt=0)
    max_output_tokens: int = Field(default=131072, gt=0)
    thinking: str | None = None


class PiSandboxedAgent(SandboxedCLIAgent):
    config: PiSandboxedAgentConfig
    observation_source = "pi"
    cli_name = "pi"

    async def execute_response(
        self, session: SandboxedAgentSession, request: Request, body: NeMoGymResponseCreateParamsNonStreaming
    ) -> NeMoGymResponse:
        prompt, system = text_prompt(body, self.config.system_prompt)
        if body.temperature is not None:
            raise ValueError(
                "Pinned Pi does not expose per-request temperature in this adapter; use model-server sampling config"
            )
        # Pi's JSON model-provider dialect is shared with the existing adapter.
        models = {
            "providers": {
                "nemo": {
                    "baseUrl": self.resolve_model_base_url(
                        self.config.model_server.name, session.seed.episode_id.capture_key
                    ),
                    "api": "openai-completions",
                    "apiKey": "EMPTY",
                    "compat": {"supportsDeveloperRole": False, "supportsReasoningEffort": False},
                    "models": [
                        {
                            "id": self.config.model,
                            "reasoning": True,
                            "input": ["text"],
                            "contextWindow": self.config.context_window,
                            "maxTokens": body.max_output_tokens
                            if body.max_output_tokens is not None
                            else self.config.max_output_tokens,
                        }
                    ],
                }
            }
        }
        await self.upload_text(session, "home/.pi/agent/models.json", json.dumps(models))
        command = [
            self.cli,
            "--print",
            "--mode",
            "json",
            "--no-session",
            "--provider",
            "nemo",
            "--model",
            self.config.model,
        ]
        if self.config.thinking:
            command += ["--thinking", self.config.thinking]
        if system:
            command += ["--append-system-prompt", system]
        command += [prompt]
        summary, stdout = await self.run_cli(session, command, env={"PI_SKIP_VERSION_CHECK": "1", "PI_TELEMETRY": "0"})
        output, usage = [], {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0, "errors": []}
        for line in stdout.splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if not isinstance(event, dict) or event.get("type") != "message_end":
                continue
            message = event.get("message") or {}
            if not isinstance(message, dict):
                continue
            if message.get("role") == "assistant":
                for block in message.get("content") or []:
                    if isinstance(block, dict) and block.get("type") == "thinking" and block.get("thinking"):
                        output.append(
                            NeMoGymResponseReasoningItem(
                                id=f"reasoning-{len(output)}",
                                summary=[{"type": "summary_text", "text": block["thinking"]}],
                            )
                        )
                usage["cached_tokens"] += int((message.get("usage") or {}).get("cacheRead") or 0)
                if message.get("stopReason") == "error":
                    usage["errors"].append(message.get("errorMessage") or "Pi model error")
                elif message.get("stopReason") == "length":
                    summary["budget_exhausted"] = True
                elif message.get("stopReason") == "aborted" and not summary.get("timed_out"):
                    usage["errors"].append("Pi aborted before completion")
            items, tokens = parse_pi_events(line)
            output.extend(items)
            for key in ("input_tokens", "output_tokens"):
                usage[key] += tokens[key]
        return self.response_from_output(
            body, [item.model_dump(mode="json") for item in output], usage, summary, session=session
        )


if __name__ == "__main__":
    PiSandboxedAgent.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = PiSandboxedAgent.run_webserver()
