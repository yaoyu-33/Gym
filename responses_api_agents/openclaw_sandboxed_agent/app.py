# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import PurePosixPath
from shlex import join

from fastapi import Request
from pydantic import Field

from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.sandboxed_agent import SandboxedAgentSession
from nemo_gym.sandboxed_cli_agent import SandboxedCLIAgent, SandboxedCLIConfig, text_prompt
from nemo_gym.server_utils import is_nemo_gym_fastapi_entrypoint
from responses_api_agents.openclaw_agent.app import (
    _decode_last_json_dict_suffix,
    parse_openclaw_output,
    parse_openclaw_session_events,
    parse_openclaw_session_items,
)


class OpenClawSandboxedAgentConfig(SandboxedCLIConfig):
    cli_version: str = "2026.6.11"
    thinking: str = "off"
    context_window: int = Field(default=262144, gt=0)
    max_output_tokens: int = Field(default=131072, gt=0)


class OpenClawSandboxedAgent(SandboxedCLIAgent):
    config: OpenClawSandboxedAgentConfig
    observation_source = "openclaw"
    cli_name = "openclaw"

    async def execute_response(
        self, session: SandboxedAgentSession, request: Request, body: NeMoGymResponseCreateParamsNonStreaming
    ) -> NeMoGymResponse:
        prompt, system = text_prompt(body, self.config.system_prompt)
        if body.temperature is not None:
            raise ValueError(
                "Pinned OpenClaw temperature overrides are not supported here; configure the Gym model server"
            )
        home = session.directory + "/home"
        env = {"HOME": home, "OPENCLAW_TELEMETRY": "0", "CLAWHUB_DISABLE_TELEMETRY": "1"}
        onboard, _ = await self.run_cli(
            session,
            [self.cli, "onboard", "--non-interactive", "--accept-risk", "--skip-health"],
            env=env,
            timeout_s=120,
        )
        if onboard.get("return_code") or onboard.get("timed_out"):
            raise RuntimeError("OpenClaw onboard failed; inspect the session stderr.log")
        config = await self.download_json(session, "home/.openclaw/openclaw.json")
        defaults = config.setdefault("agents", {}).setdefault("defaults", {})
        defaults["workspace"] = session.workdir
        for configured_agent in config["agents"].get("list", []):
            configured_agent["workspace"] = session.workdir
        defaults["model"] = {"primary": f"nemo/{self.config.model}"}
        config.setdefault("models", {}).setdefault("providers", {})["nemo"] = {
            "api": "openai-completions",
            "baseUrl": self.resolve_model_base_url(self.config.model_server.name, session.seed.episode_id.capture_key),
            "apiKey": "EMPTY",
            "models": [
                {
                    "id": self.config.model,
                    "name": self.config.model,
                    "api": "openai-completions",
                    "reasoning": True,
                    "input": ["text"],
                    "contextWindow": self.config.context_window,
                    "maxTokens": body.max_output_tokens
                    if body.max_output_tokens is not None
                    else self.config.max_output_tokens,
                }
            ],
        }
        config.setdefault("tools", {})["deny"] = ["message"]
        await self.upload_text(session, "home/.openclaw/openclaw.json", json.dumps(config))
        message = f"{system}\n\n{prompt}" if system else prompt
        command = [
            self.cli,
            "agent",
            "--local",
            "--json",
            "--agent",
            "main",
            "--thinking",
            self.config.thinking,
            "--timeout",
            str(max(1, int(self.config.sandbox_timeout))),
            "--model",
            f"nemo/{self.config.model}",
            "--message",
            message,
        ]
        summary, stdout = await self.run_cli(session, command, env=env)
        output, usage = parse_openclaw_output(stdout)
        envelope = _decode_last_json_dict_suffix(stdout) or {}
        session_file = ((envelope.get("meta") or {}).get("agentMeta") or {}).get("sessionFile")
        candidates = [session_file] if session_file else []
        if not candidates:
            # A timeout may never emit the final JSON envelope. Recover the
            # incrementally written transcript from this fresh, isolated HOME.
            script = (
                "import json,pathlib,sys; root=pathlib.Path(sys.argv[1]).resolve(); "
                "paths=[p for p in root.rglob('*.jsonl') if p.is_file() and p.resolve().is_relative_to(root)]; "
                "print(json.dumps([str(p) for p in sorted(paths,key=lambda p:p.stat().st_mtime,reverse=True)[:32]]))"
            )
            found = await self.exec_in_session(session, join([self.config.runtime_python, "-I", "-c", script, home]))
            if found.return_code:
                raise RuntimeError("Could not recover OpenClaw session transcripts")
            candidates = json.loads(found.stdout)
        events = []
        for session_file in candidates:
            relative = PurePosixPath(session_file).relative_to(PurePosixPath(session.directory))
            if ".." in relative.parts or not relative.parts or relative.parts[0] != "home":
                raise ValueError("OpenClaw transcript must remain inside the agent session home")
            events = parse_openclaw_session_events(await self.download_text(session, str(relative)))
            # The richer native parser preserves reasoning as well as tools.
            items = parse_openclaw_session_items(events, include_input=True)
            items = [item for item in items if getattr(item, "role", None) not in ("user", "system", "developer")]
            if not items:
                continue
            output = items
            if not envelope:
                usage = {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0}
                for event in events:
                    message = event.get("message") or {}
                    if message.get("role") == "assistant":
                        tokens = message.get("usage") or {}
                        usage["input_tokens"] += int(tokens.get("input") or 0) + int(tokens.get("cacheRead") or 0)
                        usage["output_tokens"] += int(tokens.get("output") or 0)
                        usage["cached_tokens"] += int(tokens.get("cacheRead") or 0)
            break
        meta = envelope.get("meta") or {}
        if meta.get("error"):
            usage.setdefault("errors", []).append(str(meta["error"]))
        terminal = next(
            (
                event["message"]
                for event in reversed(events)
                if isinstance(event.get("message"), dict) and event["message"].get("role") == "assistant"
            ),
            {},
        )
        stop_reason = meta.get("stopReason") or terminal.get("stopReason")
        if stop_reason == "length":
            summary["budget_exhausted"] = True
        elif stop_reason == "error":
            usage.setdefault("errors", []).append(terminal.get("errorMessage") or "OpenClaw model error")
        elif (meta.get("aborted") or stop_reason == "aborted") and not summary.get("timed_out"):
            usage.setdefault("errors", []).append("OpenClaw aborted before completion")
        return self.response_from_output(
            body, [item.model_dump(mode="json") for item in output], usage, summary, session=session
        )


if __name__ == "__main__":
    OpenClawSandboxedAgent.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = OpenClawSandboxedAgent.run_webserver()
