# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import yaml

from environment_servers.single_agent.app import SingleAgentEnvironmentServerConfig
from resources_servers.swebench_pro.app import SWEBenchProResourcesServerConfig
from responses_api_agents.hermes_sandboxed_agent.app import HermesSandboxedAgentConfig


def test_native_config_has_resolvable_resources_and_required_environment_limits():
    root = Path(__file__).resolve().parents[3]
    config = yaml.safe_load((Path(__file__).parents[1] / "configs/hermes_sandboxed_agent.yaml").read_text())
    resources = yaml.safe_load((root / config["config_paths"][0]).read_text())
    environment = SingleAgentEnvironmentServerConfig.model_validate({
        "name": "swe_pro_hermes", "host": "127.0.0.1", "port": 8000,
        **config["swe_pro_hermes"]["environment_servers"]["single_agent"],
    })
    assert environment.resources_server.name in resources
    resource = SWEBenchProResourcesServerConfig.model_validate({
        "name": environment.resources_server.name, "host": "127.0.0.1", "port": 8001,
        **resources[environment.resources_server.name]["resources_servers"]["swebench_pro"],
    })
    agent = HermesSandboxedAgentConfig.model_validate({
        "name": environment.agent_server.name, "host": "127.0.0.1", "port": 8002,
        **config[environment.agent_server.name]["responses_api_agents"]["hermes_sandboxed_agent"],
        "model": "super-3.5",
    })
    assert not resource.is_verifying_golden_patch
    assert resource.apply_anti_cheating
    assert environment.default_episode_timeout_seconds > agent.sandbox_timeout
