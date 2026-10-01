# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from omegaconf import OmegaConf

from nemo_gym.global_config import GlobalConfigDictParser
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient
from resources_servers.gdpval.app import GDPValResourcesServer, GDPValVerifyRequest
from resources_servers.gdpval.sandbox_app import GDPSandboxConfig
from resources_servers.gdpval.sandbox_tasks import prepare_row
from resources_servers.gdpval.tests.live_sandbox_smoke import GenerationOnlyResources, make_agent


@pytest.mark.parametrize("harness", ["hermes", "pi"])
def test_pairing_config_uses_shared_gdp_lifecycle(harness: str) -> None:
    _, layers = GlobalConfigDictParser().load_extra_config_paths(
        [f"resources_servers/gdpval/configs/gdpval_{harness}_sandbox.yaml"]
    )
    config = OmegaConf.merge(*layers, {"policy_model_name": "test-policy"})
    resources = config.gdpval_resources_server.resources_servers.gdpval
    agent_name = f"{harness}_agent"
    environment = config.single_agent_turn_legacy.environment_servers.single_agent_turn_legacy
    assert resources.entrypoint == "sandbox_app.py"
    assert list(resources.allowed_agents) == [agent_name]
    assert resources.judge_model_server.name == "gdpval_judge_model"
    assert resources.verified is False
    assert environment.agent_server.name == agent_name
    assert environment.resources_server.name == "gdpval_resources_server"
    assert environment.max_concurrent_episodes == 1
    assert config.sandbox.apptainer.create.mount_point == "/workspace"
    assert ("pi_agent" in config) == (harness == "pi")
    assert ("hermes_agent" in config) == (harness == "hermes")
    if harness == "hermes":
        agent = config.hermes_agent.responses_api_agents.hermes_agent
        assert agent.model == "test-policy"
        assert agent.resources_server.name == "gdpval_resources_server"
        assert list(agent.enabled_toolsets) == ["terminal", "file"]
        assert agent.sandbox_provider is None  # Borrow the task sandbox; do not create another.


@pytest.mark.parametrize("harness", ["hermes", "pi"])
def test_smoke_uses_existing_agent_with_requested_budget(harness: str) -> None:
    if harness == "hermes":
        pytest.importorskip("model_tools", reason="Install Hermes agent requirements for this check")
    args = argparse.Namespace(harness=harness, model="test-policy", max_output_tokens=16384, input=Path("tasks.jsonl"))
    agent = make_agent(
        args,
        {"name": f"{harness}_agent", "host": "127.0.0.1", "port": 12345, "entrypoint": "app.py"},
        MagicMock(spec=ServerClient),
    )
    assert agent.config.model == "test-policy"
    assert agent.config.model_server.name == "policy_model"
    if harness == "hermes":
        assert agent.__class__.__name__ == "HermesAgent"
        assert agent.config.max_tokens == 16384
        assert agent.config.sandbox_provider is None
    else:
        assert agent.__class__.__name__ == "PiAgent"
        assert agent.config.max_output_tokens == 16384


async def test_generation_smoke_masks_reward_and_never_calls_judge(tmp_path: Path, monkeypatch) -> None:
    judge = AsyncMock(side_effect=AssertionError("Smoke must not call the live judge"))
    monkeypatch.setattr(GDPValResourcesServer, "verify", judge)
    export = AsyncMock(return_value=tmp_path)
    monkeypatch.setattr(GenerationOnlyResources, "export_deliverables", export)
    instance = GenerationOnlyResources(
        config=GDPSandboxConfig(
            name="resources",
            host="127.0.0.1",
            port=12345,
            entrypoint="sandbox_app.py",
            image="test-image",
            deliverables_root=tmp_path,
            preconvert_office_to_pdf=False,
            judge_model_server={"type": "responses_api_models", "name": "unused-judge"},
        ),
        server_client=MagicMock(spec=ServerClient),
    )
    source = json.loads(Path("resources_servers/gdpval/data/example.jsonl").read_text().splitlines()[0])
    prepared = prepare_row(source)
    body = GDPValVerifyRequest.model_validate(
        prepared
        | {
            "response": NeMoGymResponse(
                id="response",
                object="response",
                created_at=0,
                model="test-policy",
                output=[],
                tools=[],
                tool_choice="auto",
                parallel_tool_calls=False,
            ).model_dump()
        }
    )
    result = await instance.verify(SimpleNamespace(session={SESSION_ID_KEY: "session"}), body)
    export.assert_awaited_once_with("session")
    judge.assert_not_awaited()
    assert result.mask_sample is True
    assert "not a score" in result.failure_reason
    assert result.deliverables_dir == str(tmp_path)
