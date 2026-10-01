# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in real-model/Docker file-task smoke, NOT a GDP benchmark or judge run.

python -m resources_servers.gdpval.tests.live_pi_contract_smoke \
    --model-url http://MODEL:8000/v1 --model n35-super-ga \
    --host 10.111.115.167 --image EXISTING_LINUX_IMAGE --output /absolute/new/run

Only the local CSV reference source and judge are replaced. Native Environment,
Resources and Pi session endpoints, model proxy, sandbox tools/export/cleanup are real.
"""

import argparse
import asyncio
import csv
import json
import os
import socket
from pathlib import Path
from uuid import uuid4

import uvicorn
from aiohttp import ClientTimeout
from fastapi import Request
from omegaconf import OmegaConf

import nemo_gym.global_config as global_config
from environment_servers.single_agent_turn.app import (
    SingleAgentTurnEnvironmentServer,
    SingleAgentTurnEnvironmentServerConfig,
)
from nemo_gym.sandbox import AsyncSandbox
from nemo_gym.server_utils import SESSION_ID_KEY, BaseServerConfig, ServerClient
from nemo_gym.server_utils import request as http_request
from resources_servers.gdpval.app import GDPValVerifyRequest, GDPValVerifyResponse
from resources_servers.gdpval.sandbox_app import GDPSandboxConfig, GDPSandboxResourcesServer
from resources_servers.gdpval.sandbox_tasks import GDPFileTask, prepare_row
from responses_api_agents.pi_agent.app import PiAgent, PiAgentConfig
from responses_api_models.openai_model.app import SimpleModelServer, SimpleModelServerConfig


class ExportOnlyResources(GDPSandboxResourcesServer):
    """Test-only local reference and no-judge boundary; production scorer is unchanged."""

    async def _stage_references(self, sandbox: AsyncSandbox, task: GDPFileTask) -> None:
        source = self.config.deliverables_root / "reference.csv"
        await sandbox.upload(source, "/workspace/input/reference.csv")

    async def verify(self, request: Request, body: GDPValVerifyRequest) -> GDPValVerifyResponse:
        directory = await self.export_deliverables(request.session[SESSION_ID_KEY])
        return GDPValVerifyResponse(
            **(body.model_dump() | {"deliverables_dir": str(directory)}),
            reward=0.0,
            mask_sample=True,
            failure_reason="Contract smoke only: judge deliberately not run; reward placeholder is not a score.",
        )


async def run(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "reference.csv").write_text("item,amount\nalpha,1\nalpha,5\nbeta,9\n")
    sockets = []
    for _ in range(4):
        sock = socket.socket()
        sock.bind(("0.0.0.0", 0))
        sockets.append(sock)
    names = ["resources", "pi_agent", "policy_model", "environment"]
    common = {
        name: {"host": args.host, "port": sock.getsockname()[1], "name": name, "entrypoint": "app.py"}
        for name, sock in zip(names, sockets, strict=True)
    }
    cfg = OmegaConf.create(
        {
            "resources": {"resources_servers": {"gdpval": common["resources"]}},
            "pi_agent": {"responses_api_agents": {"pi_agent": common["pi_agent"]}},
            "policy_model": {"responses_api_models": {"openai_model": common["policy_model"]}},
            "environment": {"environment_servers": {"single_agent_turn": common["environment"]}},
            "sandbox": {"docker": {"create": {"use_init": True}}},
        }
    )
    global_config._GLOBAL_CONFIG_DICT = cfg
    client = ServerClient(head_server_config=BaseServerConfig(host="127.0.0.1", port=1), global_config_dict=cfg)
    resources = ExportOnlyResources(
        config=GDPSandboxConfig(
            **common["resources"],
            image=args.image,
            deliverables_root=args.output,
            preconvert_office_to_pdf=False,
            judge_model_server={"type": "responses_api_models", "name": "policy_model"},
        ),
        server_client=client,
    )
    pi = PiAgent(
        config=PiAgentConfig(
            **common["pi_agent"],
            model=args.model,
            pi_version="0.80.2",
            model_server={"type": "responses_api_models", "name": "policy_model"},
            timeout=300,
            context_window=262144,
            max_output_tokens=4096,
            thinking="low",
        ),
        server_client=client,
    )
    model = SimpleModelServer(
        config=SimpleModelServerConfig(
            **common["policy_model"],
            openai_base_url=args.model_url,
            openai_model=args.model,
            openai_api_key=os.environ.get("PI_SMOKE_MODEL_KEY", "dummy"),
            max_http_attempts=1,
        ),
        server_client=client,
    )
    environment = SingleAgentTurnEnvironmentServer(
        config=SingleAgentTurnEnvironmentServerConfig(
            **common["environment"],
            resources_server={"type": "resources_servers", "name": "resources"},
            agent_server={"type": "responses_api_agents", "name": "pi_agent"},
            default_episode_timeout_seconds=900,
            cleanup_timeout_seconds=120,
        ),
        server_client=client,
    )
    servers = [
        uvicorn.Server(uvicorn.Config(component.setup_webserver(), log_level="info"))
        for component in (resources, pi, model, environment)
    ]
    tasks = [asyncio.create_task(server.serve(sockets=[sock])) for server, sock in zip(servers, sockets, strict=True)]
    try:
        async with asyncio.timeout(30):
            while not all(server.started for server in servers):
                if any(task.done() for task in tasks):
                    raise RuntimeError("A smoke server failed at startup")
                await asyncio.sleep(0.1)
        row = prepare_row(
            {
                "task_id": "synthetic-csv-contract",
                "prompt": "Read reference.csv and sum amount by item. Submit summary.csv with columns item,total. "
                "Include one row for each item, sorted alphabetically. Do not copy the reference as output.",
                "reference_files": ["reference.csv"],
                "reference_file_urls": ["https://example.invalid/local-test-fixture"],
            }
        )
        params = row.pop("responses_create_params")
        body = {
            "episode_id": {"rollout_id": f"pi-gdp-contract-{uuid4().hex}"},
            "task": {
                "task_id": {"taskset": "gdp-contract", "task_id": row["task_id"]},
                "task_input": {"responses_create_params": params, "task_data": row},
            },
        }
        (args.output / "request.json").write_text(json.dumps(body, indent=2))
        url = f"http://127.0.0.1:{common['environment']['port']}/run"
        reply = await http_request("POST", url, json=body, timeout=ClientTimeout(total=1200))
        try:
            reply.raise_for_status()
            result = await reply.json()
        finally:
            reply.release()
        (args.output / "episode.json").write_text(json.dumps(result, indent=2))
        if result.get("failure"):
            raise RuntimeError(json.dumps(result["failure"]))
        verdict = result["result"]
        assert verdict["mask_sample"] is True
        output = Path(verdict["deliverables_dir"]) / "summary.csv"
        with output.open() as stream:
            values = {record["item"]: float(record["total"]) for record in csv.DictReader(stream)}
        assert values == {"alpha": 6, "beta": 9}, values
        assert not resources._sessions, "Resources session was not closed"
        summary = {
            "contract_smoke": "passed",
            "benchmark": False,
            "judge_run": False,
            "reward": None,
            "deliverables": str(output),
            "csv_values": values,
            "usage": verdict["response"].get("usage"),
            "resources_closed": True,
        }
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary), flush=True)
    finally:
        for server in servers:
            server.should_exit = True
        await asyncio.gather(*tasks, return_exceptions=True)
        for sock in sockets:
            sock.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--host", required=True, help="Host IP reachable from task containers")
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True, type=Path)
    asyncio.run(run(parser.parse_args()))
