# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline demo: choose an environment once, without editing existing YAML files.

Run: .venv/bin/python scripts/demo_auto_environment_routing.py

This is NOT wired into Gym startup and does not start servers or certify pairings.
The proposed insertion point is after config composition and before server startup.
The readiness callback represents an explicit, validated native-support declaration
for the complete configured pairing; endpoint presence alone is not sufficient.

Existing flat rows need `single_agent_legacy` to translate their input/output shape.
Despite its name, that adapter executes the NEW lifecycle, not the agent's old /run.
Unmigrated pairs use `legacy_agent`, which forwards to the old agent /run.

The collector would use the returned route for BOTH /run and /aggregate_metrics,
including retries. There is no per-rollout capability probe or failure fallback.
The upstream adapters, native aggregation route, and collector wiring are separate
prerequisites; this demo deliberately tests only startup selection/config synthesis.
"""

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, Mapping

from omegaconf import DictConfig, OmegaConf


@dataclass(frozen=True)
class EnvironmentRoute:
    """A startup decision shared by execution, retries, and metrics aggregation."""

    server_name: str
    server_type: str
    selection: Literal["native", "compatibility", "explicit"]


def _component(instance: DictConfig, key: str) -> tuple[str, DictConfig] | None:
    components = instance.get(key)
    if components is None:
        return None
    if not isinstance(components, DictConfig) or len(components) != 1:
        raise ValueError(f"Expected exactly one component under {key}")
    kind = next(iter(components))
    settings = components[kind]
    if not isinstance(settings, DictConfig):
        raise ValueError(f"Expected settings for {key}.{kind}")
    return str(kind), settings


def resolve_environment_routes(
    config: DictConfig,
    *,
    supports_native: Callable[[DictConfig, DictConfig], bool],
) -> tuple[DictConfig, Mapping[str, EnvironmentRoute]]:
    """Copy composed config and synthesize missing single-agent environment bindings.

    `supports_native` receives complete agent/resources instances and must be a
    read-only startup check for the default single-agent protocol (no resource
    tool transports). False means explicitly unsupported, not unreachable.
    Exceptions are startup errors, never a reason to silently choose compatibility.
    Explicit environment bindings take precedence. Multiple environments for one
    agent require explicit taskset routing and are outside this small demo.
    """
    resolved = deepcopy(config)
    routes: dict[str, EnvironmentRoute] = {}

    for name, instance in resolved.items():
        if not isinstance(instance, DictConfig):
            continue
        environment = _component(instance, "environment_servers")
        if environment is None:
            continue
        kind, settings = environment
        agent_name = OmegaConf.select(settings, "agent_server.name")
        if not isinstance(agent_name, str) or not agent_name:
            raise ValueError(f"{name}: this demo requires a single-agent environment")
        agent_instance = resolved.get(agent_name)
        if not isinstance(agent_instance, DictConfig) or _component(agent_instance, "responses_api_agents") is None:
            raise ValueError(f"{name}: missing agent server {agent_name!r}")
        if agent_name in routes:
            raise ValueError(f"{agent_name}: multiple environments require explicit taskset routing")
        routes[agent_name] = EnvironmentRoute(str(name), kind, "explicit")

    for agent_name, instance in list(resolved.items()):
        if not isinstance(instance, DictConfig):
            continue
        agent = _component(instance, "responses_api_agents")
        if agent is None or agent_name in routes:
            continue
        _, settings = agent
        reference = settings.get("resources_server")
        if isinstance(reference, DictConfig) and OmegaConf.is_missing(reference, "name"):
            continue  # Unbound swap template, not an executable pairing.

        resources_name = None
        native = False
        if reference is not None:
            if not isinstance(reference, DictConfig) or reference.get("type") != "resources_servers":
                raise ValueError(f"{agent_name}: invalid resources_server reference")
            resources_name = reference.get("name")
            if not isinstance(resources_name, str) or not resources_name:
                raise ValueError(f"{agent_name}: resources_server.name is required")
            resources = resolved.get(resources_name)
            if not isinstance(resources, DictConfig) or _component(resources, "resources_servers") is None:
                raise ValueError(f"{agent_name}: missing resources server {resources_name!r}")
            native = supports_native(instance, resources)
            if not isinstance(native, bool):
                raise TypeError(f"{agent_name}: native readiness must be explicitly True or False")

        environment_name = f"{agent_name}_auto_environment"
        if environment_name in resolved:
            raise ValueError(f"Generated environment name {environment_name!r} is already in use")
        kind = "single_agent_legacy" if native else "legacy_agent"
        environment_config = {
            "entrypoint": "app.py",
            "agent_server": {"type": "responses_api_agents", "name": agent_name},
        }
        if native:
            environment_config["resources_server"] = {"type": "resources_servers", "name": resources_name}
        resolved[environment_name] = {"environment_servers": {kind: environment_config}}
        routes[str(agent_name)] = EnvironmentRoute(environment_name, kind, "native" if native else "compatibility")

    return resolved, MappingProxyType(routes)


def main() -> None:
    """Print three synthetic decisions; never connect to a model or sandbox."""
    config = OmegaConf.create(
        {
            "swe": {"resources_servers": {"swebench_pro": {"entrypoint": "app.py"}}},
            "math": {"resources_servers": {"math": {"entrypoint": "app.py"}}},
            **{
                name: {
                    "responses_api_agents": {
                        kind: {
                            "entrypoint": "app.py",
                            "num_workers": 1,
                            "resources_server": {"type": "resources_servers", "name": resource},
                        }
                    }
                }
                for name, kind, resource in (
                    ("hermes_swe", "hermes_agent", "swe"),
                    ("opencode_swe", "opencode_agent", "swe"),
                    ("hermes_math", "hermes_agent", "math"),
                )
            },
        }
    )

    def demo_supports_native(agent: DictConfig, resources: DictConfig) -> bool:
        # Illustrative opt-in for these synthetic fixtures ONLY, not a production
        # readiness check. Real declarations must cover settings, tools and sandbox needs.
        return "hermes_agent" in agent.responses_api_agents and "swebench_pro" in resources.resources_servers

    resolved, routes = resolve_environment_routes(config, supports_native=demo_supports_native)
    print("DEMO ONLY: synthetic readiness declaration; no servers launched or YAML files changed.")
    for agent_name, route in routes.items():
        print(f"{agent_name}: {route.selection} -> {route.server_name} ({route.server_type})")
    print("\nGenerated in-memory environment config:")
    print(
        OmegaConf.to_yaml(
            OmegaConf.create({route.server_name: resolved[route.server_name] for route in routes.values()})
        )
    )


if __name__ == "__main__":
    main()
