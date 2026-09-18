# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import FrozenInstanceError
from unittest.mock import Mock

import pytest
from omegaconf import DictConfig, OmegaConf
from scripts.demo_auto_environment_routing import main, resolve_environment_routes


def _config() -> DictConfig:
    return OmegaConf.create(
        {
            "policy": {"responses_api_models": {"vllm_model": {"entrypoint": "app.py"}}},
            "swe": {"resources_servers": {"swebench_pro": {"entrypoint": "app.py"}}},
            "agent": {
                "responses_api_agents": {
                    "hermes_agent": {
                        "entrypoint": "app.py",
                        "num_workers": 1,
                        "model_server": {"type": "responses_api_models", "name": "policy"},
                        "resources_server": {"type": "resources_servers", "name": "swe"},
                        "datasets": [{"name": "example", "jsonl_fpath": "unchanged.jsonl"}],
                    }
                }
            },
        }
    )


@pytest.mark.parametrize("native", [True, False])
def test_synthesizes_route_without_changing_existing_config(native: bool) -> None:
    config = _config()
    before = OmegaConf.to_yaml(config)
    check = Mock(return_value=native)

    resolved, routes = resolve_environment_routes(config, supports_native=check)

    assert OmegaConf.to_yaml(config) == before
    for name in config:
        assert resolved[name] == config[name]
    check.assert_called_once_with(config.agent, config.swe)
    route = routes["agent"]
    kind = "single_agent_legacy" if native else "legacy_agent"
    assert route.selection == ("native" if native else "compatibility")
    assert route.server_type == kind
    generated = resolved[route.server_name].environment_servers[kind]
    assert generated.agent_server == {"type": "responses_api_agents", "name": "agent"}
    if native:
        assert generated.resources_server == {"type": "resources_servers", "name": "swe"}
    else:
        assert "resources_server" not in generated


def test_native_support_is_for_the_complete_configured_pairing() -> None:
    config = _config()
    config.math = {"resources_servers": {"math": {"entrypoint": "app.py"}}}
    config.other_agent = OmegaConf.create(OmegaConf.to_container(config.agent))
    config.other_agent.responses_api_agents.hermes_agent.resources_server.name = "math"

    def supports_native(agent: DictConfig, resources: DictConfig) -> bool:
        return (
            "hermes_agent" in agent.responses_api_agents
            and agent.responses_api_agents.hermes_agent.num_workers == 1
            and "swebench_pro" in resources.resources_servers
        )

    _, routes = resolve_environment_routes(config, supports_native=supports_native)
    assert routes["agent"].selection == "native"
    assert routes["other_agent"].selection == "compatibility"

    config.agent.responses_api_agents.hermes_agent.num_workers = 2
    _, new_routes = resolve_environment_routes(config, supports_native=supports_native)
    assert new_routes["agent"].selection == "compatibility"
    assert routes["agent"].selection == "native"  # An existing run keeps its startup decision.


def test_explicit_environment_wins_and_resolution_is_idempotent() -> None:
    resolved, first_routes = resolve_environment_routes(_config(), supports_native=Mock(return_value=True))
    check = Mock(side_effect=AssertionError("Explicit bindings must not be auto-switched"))

    second, routes = resolve_environment_routes(resolved, supports_native=check)

    assert second == resolved
    assert routes["agent"].selection == "explicit"
    assert routes["agent"].server_name == first_routes["agent"].server_name
    assert routes["agent"].server_type == first_routes["agent"].server_type
    check.assert_not_called()


def test_missing_capability_is_not_inferred_from_a_session_endpoint() -> None:
    config = _config()
    # Even an agent advertising this URL needs explicit native-readiness approval.
    config.agent.responses_api_agents.hermes_agent.session_url = "/v1/agent_sessions"
    _, routes = resolve_environment_routes(config, supports_native=Mock(return_value=False))
    assert routes["agent"].selection == "compatibility"


def test_explicit_compatibility_is_not_upgraded_automatically() -> None:
    resolved, _ = resolve_environment_routes(_config(), supports_native=Mock(return_value=False))
    check = Mock(return_value=True)
    _, routes = resolve_environment_routes(resolved, supports_native=check)
    assert routes["agent"].selection == "explicit"
    assert routes["agent"].server_type == "legacy_agent"
    check.assert_not_called()


@pytest.mark.parametrize("readiness", [None, {}, "true"])
def test_unknown_readiness_is_an_error_not_a_compatibility_decision(readiness: object) -> None:
    with pytest.raises(TypeError, match="explicitly True or False"):
        resolve_environment_routes(_config(), supports_native=Mock(return_value=readiness))


def test_failed_readiness_check_is_not_silent_fallback() -> None:
    config = _config()
    before = OmegaConf.to_yaml(config)
    with pytest.raises(ConnectionError, match="unreachable"):
        resolve_environment_routes(config, supports_native=Mock(side_effect=ConnectionError("unreachable")))
    assert OmegaConf.to_yaml(config) == before


def test_routes_cannot_be_mutated_after_startup() -> None:
    _, routes = resolve_environment_routes(_config(), supports_native=Mock(return_value=True))
    with pytest.raises(TypeError):
        routes["agent"] = routes["agent"]
    with pytest.raises(FrozenInstanceError):
        routes["agent"].server_name = "another_server"
    with pytest.raises(KeyError):
        _ = routes["unknown_agent"]  # No implicit fallthrough to direct agent /run.


def test_multiple_environments_do_not_silently_pick_the_first() -> None:
    resolved, _ = resolve_environment_routes(_config(), supports_native=Mock(return_value=True))
    resolved.another_environment = OmegaConf.create(OmegaConf.to_container(resolved.agent_auto_environment))
    with pytest.raises(ValueError, match="multiple environments require explicit taskset routing"):
        resolve_environment_routes(resolved, supports_native=Mock(return_value=True))


def test_name_collision_does_not_overwrite_existing_config() -> None:
    config = _config()
    config.agent_auto_environment = {"user_setting": "keep me"}
    with pytest.raises(ValueError, match="already in use"):
        resolve_environment_routes(config, supports_native=Mock(return_value=True))
    assert config.agent_auto_environment.user_setting == "keep me"


@pytest.mark.parametrize("reference", [{"type": "resources_servers", "name": "absent"}, {}, "invalid"])
def test_invalid_resource_binding_fails_instead_of_falling_back(reference: object) -> None:
    config = _config()
    config.agent.responses_api_agents.hermes_agent.resources_server = reference
    with pytest.raises(ValueError, match="resources"):
        resolve_environment_routes(config, supports_native=Mock(return_value=False))


def test_unbound_templates_are_not_started() -> None:
    config = _config()
    config.agent.responses_api_agents.hermes_agent.resources_server.name = "???"
    check = Mock()
    resolved, routes = resolve_environment_routes(config, supports_native=check)
    assert resolved == config
    assert not routes
    check.assert_not_called()


def test_explicit_environment_cannot_reference_a_missing_agent() -> None:
    resolved, _ = resolve_environment_routes(_config(), supports_native=Mock(return_value=True))
    del resolved.agent
    with pytest.raises(ValueError, match="missing agent server"):
        resolve_environment_routes(resolved, supports_native=Mock(return_value=True))


def test_self_contained_old_agent_needs_no_resources_binding() -> None:
    config = _config()
    del config.agent.responses_api_agents.hermes_agent.resources_server
    check = Mock()
    _, routes = resolve_environment_routes(config, supports_native=check)
    assert routes["agent"].selection == "compatibility"
    check.assert_not_called()


def test_demo_prints_native_and_compatibility_choices(capsys: pytest.CaptureFixture[str]) -> None:
    main()
    output = capsys.readouterr().out
    assert "hermes_swe: native" in output
    assert "opencode_swe: compatibility" in output
    assert "hermes_math: compatibility" in output
    assert "no servers launched or YAML files changed" in output
