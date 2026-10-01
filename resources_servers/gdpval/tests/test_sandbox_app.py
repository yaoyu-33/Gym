# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from nemo_gym.base_resources_server import ResourcesCloseSessionRequest, ResourcesSeedSessionRequest
from nemo_gym.episode_types import EpisodeId, TaskId
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient
from resources_servers.gdpval import sandbox_app
from resources_servers.gdpval.app import GDPValResourcesServer, GDPValVerifyRequest, GDPValVerifyResponse
from resources_servers.gdpval.sandbox_app import GDPSandboxConfig, GDPSandboxResourcesServer
from resources_servers.gdpval.sandbox_tasks import GDPFileTask, prepare_row


def row(**extra):
    return {
        "task_id": "task-1",
        "prompt": "Write a financial summary.",
        "responses_create_params": {"input": []},
        "reference_files": [],
        "reference_file_urls": [],
        "rubric_pretty": "PRIVATE RUBRIC",
        **extra,
    }


def seed(**extra):
    return ResourcesSeedSessionRequest(
        resources_session_id="resources-1",
        episode_id=EpisodeId(rollout_id="rollout-1"),
        task_id=TaskId(taskset="gdp", task_id="task-1"),
        task_data=row(),
        **extra,
    )


def response():
    return NeMoGymResponse(
        id="response-1",
        created_at=0,
        model="model",
        object="response",
        output=[],
        tools=[],
        tool_choice="auto",
        parallel_tool_calls=False,
    )


class Sandbox:
    def __init__(self):
        self.files = {"/workspace/output/report.csv": b"name,value\na,3\n"}
        self.start = AsyncMock()
        self.stop = AsyncMock()
        self.serialize = AsyncMock(return_value={"sandbox_id": "task-box"})
        self.exec = AsyncMock(side_effect=self.execute)
        self.upload = AsyncMock(side_effect=self.upload_file)
        self.download = AsyncMock(side_effect=self.download_file)

    async def execute(self, command, **kwargs):
        files = [
            {"name": key.removeprefix("/workspace/output/"), "size": len(value)}
            for key, value in self.files.items()
            if key.startswith("/workspace/output/")
        ]
        return SimpleNamespace(return_code=0, stdout=json.dumps(files), stderr="")

    async def upload_file(self, local, remote):
        self.files[remote] = Path(local).read_bytes()

    async def download_file(self, remote, local):
        Path(local).write_bytes(self.files[remote])


@pytest.fixture
def server(tmp_path, monkeypatch):
    box = Sandbox()
    monkeypatch.setattr(sandbox_app, "AsyncSandbox", lambda provider: box)
    monkeypatch.setattr(sandbox_app, "get_global_config_dict", lambda: {})
    monkeypatch.setattr(sandbox_app, "resolve_provider_config", lambda *args: {"docker": {}})
    config = GDPSandboxConfig(
        host="127.0.0.1",
        port=8000,
        name="resources",
        entrypoint="sandbox_app.py",
        image="test-only",
        deliverables_root=tmp_path,
        preconvert_office_to_pdf=False,
        judge_model_server={"type": "responses_api_models", "name": "judge"},
    )
    instance = GDPSandboxResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
    return instance, box, SimpleNamespace(session={})


def test_prepare_only_exposes_prompt_and_reference_paths():
    source = row(reference_files='["a.xlsx"]', reference_file_urls='["https://example.com/a.xlsx"]')
    prepared = prepare_row(source)
    text = prepared["responses_create_params"]["input"][0]["content"]
    assert "PRIVATE RUBRIC" not in text
    assert "https://example.com" not in text
    assert "/workspace/input/a.xlsx" in text
    assert "/workspace/output" in text
    assert "finish tool" not in text
    assert prepared["rubric_pretty"] == "PRIVATE RUBRIC"
    assert source["responses_create_params"]["input"] == []


@pytest.mark.parametrize("name", ["../secret", "/etc/passwd", "a/../../x", "a\\b", "a//b", ".", "a/./b", "a\x00b"])
def test_unsafe_reference_paths_rejected(name):
    with pytest.raises(ValueError):
        GDPFileTask.model_validate(row(reference_files=[name], reference_file_urls=["https://example.com/file"]))


@pytest.mark.parametrize(
    "fields",
    [
        {"reference_files": ["x"], "reference_file_urls": []},
        {"reference_files": ["x", "x"], "reference_file_urls": ["https://example.com/x"] * 2},
        {"reference_files": ["x"], "reference_file_urls": ["file:///secret"]},
    ],
)
def test_bad_reference_metadata_rejected(fields):
    with pytest.raises(ValueError):
        GDPFileTask.model_validate(row(**fields))


async def test_seed_is_idempotent_and_lends_resources_owned_sandbox(server):
    instance, box, request = server
    first, second = await asyncio.gather(
        instance.seed_session(request, seed()), instance.seed_session(request, seed())
    )
    assert first == second
    assert first.sandbox_access.workdir == "/workspace"
    assert first.sandbox_access.connection.descriptor == {"sandbox_id": "task-box"}
    assert request.session[SESSION_ID_KEY] == "resources-1"
    box.start.assert_awaited_once()
    box.stop.assert_not_awaited()
    spec = box.start.call_args.args[0]
    assert "PRIVATE RUBRIC" not in str(spec)


async def test_seed_conflict_rejected(server):
    instance, _, request = server
    body = seed()
    await instance.seed_session(request, body)
    body.task_data["prompt"] = "Different task"
    with pytest.raises(HTTPException, match="bound"):
        await instance.seed_session(request, body)


async def test_reference_bytes_staged_before_sandbox_is_exposed(server, monkeypatch):
    instance, box, request = server
    content = b"PK\x00\xffbinary spreadsheet\n"

    async def chunks(*args):
        yield content

    download = SimpleNamespace(
        raise_for_status=MagicMock(),
        release=MagicMock(),
        content=SimpleNamespace(iter_chunked=chunks),
    )
    monkeypatch.setattr(sandbox_app, "http_request", AsyncMock(return_value=download))
    body = seed()
    body.task_data.update(reference_files=["nested/a.xlsx"], reference_file_urls=["https://example.com/a"])
    await instance.seed_session(request, body)
    assert box.files["/workspace/input/nested/a.xlsx"] == content
    download.release.assert_called_once()


async def test_reference_failure_stops_sandbox_and_never_exposes_access(server, monkeypatch):
    instance, box, request = server
    monkeypatch.setattr(instance, "_stage_references", AsyncMock(side_effect=RuntimeError("download failed")))
    with pytest.raises(RuntimeError, match="download failed"):
        await instance.seed_session(request, seed())
    box.stop.assert_awaited_once()
    box.serialize.assert_not_awaited()
    assert SESSION_ID_KEY not in request.session


async def test_verify_exports_bytes_and_reuses_existing_gdp_judge(server, monkeypatch):
    instance, box, request = server
    await instance.seed_session(request, seed())
    captured = []

    async def grade(self, body):
        captured.append(body)
        assert Path(body.deliverables_dir, "report.csv").read_bytes() == b"name,value\na,3\n"
        assert body.rubric_pretty == "PRIVATE RUBRIC"
        return GDPValVerifyResponse(**body.model_dump(), reward=0.75)

    monkeypatch.setattr(GDPValResourcesServer, "verify", grade)
    body = GDPValVerifyRequest(
        **row(rubric_pretty="TAMPERED"), response=response(), deliverables_dir="/untrusted/path"
    )
    first = await instance.verify(request, body)
    second = await instance.verify(request, body)
    assert first.reward == second.reward == 0.75
    assert len(captured) == 1
    box.stop.assert_not_awaited()
    close = ResourcesCloseSessionRequest(resources_session_id="resources-1", episode_id=seed().episode_id)
    assert await instance.close_resources_session(request, close) == await instance.close_resources_session(
        request, close
    )
    box.stop.assert_awaited_once()
    assert Path(first.deliverables_dir, "report.csv").exists()
    with pytest.raises(HTTPException, match="closed"):
        await instance.seed_session(request, seed())


async def test_invalid_judge_is_retryable_not_zero_reward(server, monkeypatch):
    instance, _, request = server
    await instance.seed_session(request, seed())

    async def invalid(self, body):
        return GDPValVerifyResponse(**body.model_dump(), reward=0.0, invalid_judge_response=True)

    monkeypatch.setattr(GDPValResourcesServer, "verify", invalid)
    with pytest.raises(HTTPException) as error:
        await instance.verify(request, GDPValVerifyRequest(**row(), response=response()))
    assert error.value.status_code == 503
    assert instance._sessions["resources-1"].verdict is None


async def test_failed_close_retains_handle_for_retry(server):
    instance, box, request = server
    await instance.seed_session(request, seed())
    box.stop.side_effect = [RuntimeError("provider unavailable"), None]
    close = ResourcesCloseSessionRequest(resources_session_id="resources-1", episode_id=seed().episode_id)
    with pytest.raises(RuntimeError):
        await instance.close_resources_session(request, close)
    assert "resources-1" in instance._sessions
    await instance.close_resources_session(request, close)
    assert "resources-1" not in instance._sessions
    assert box.stop.await_count == 2


async def test_close_before_seed_fences_delayed_seed(server):
    instance, box, request = server
    close = ResourcesCloseSessionRequest(resources_session_id="resources-1", episode_id=seed().episode_id)
    await instance.close_resources_session(request, close)
    with pytest.raises(HTTPException):
        await instance.seed_session(request, seed())
    box.start.assert_not_awaited()


async def test_export_failure_prevents_grading(server, monkeypatch):
    instance, box, request = server
    await instance.seed_session(request, seed())
    box.exec.side_effect = None
    box.exec.return_value = SimpleNamespace(return_code=1, stderr="symlink rejected")
    grader = AsyncMock()
    monkeypatch.setattr(GDPValResourcesServer, "verify", grader)
    with pytest.raises(HTTPException) as error:
        await instance.verify(request, GDPValVerifyRequest(**row(), response=response()))
    assert error.value.status_code == 503
    grader.assert_not_awaited()


async def test_other_session_cannot_verify_or_close(server):
    instance, _, request = server
    await instance.seed_session(request, seed())
    with pytest.raises(HTTPException):
        await instance.verify(SimpleNamespace(session={}), GDPValVerifyRequest(**row(), response=response()))
    with pytest.raises(HTTPException):
        await instance.close_resources_session(
            request,
            ResourcesCloseSessionRequest(
                resources_session_id="resources-1",
                episode_id=EpisodeId(rollout_id="other"),
            ),
        )


def test_config_rejects_relative_output_and_multiple_workers(server):
    instance, _, _ = server
    for change in ({"deliverables_root": "relative"}, {"num_workers": 2}):
        with pytest.raises(ValidationError):
            GDPSandboxConfig.model_validate(instance.config.model_dump() | change)


@pytest.mark.parametrize("kind", ["file", "symlink", "directory", "hardlink"])
def test_actual_export_listing_rejects_nonregular_deliverables(tmp_path, kind):
    output = tmp_path / "output"
    output.mkdir()
    candidate = output / "report.csv"
    outside = tmp_path / "not-a-deliverable"
    outside.write_bytes(b"private data")
    if kind == "file":
        candidate.write_bytes(b"a,b\n1,2\n")
    elif kind == "symlink":
        candidate.symlink_to(outside)
    elif kind == "directory":
        candidate.mkdir()
    else:
        candidate.hardlink_to(outside)
    script = sandbox_app._LIST_OUTPUTS.replace(repr("/workspace/output"), repr(str(output)))
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    if kind == "file":
        assert result.returncode == 0
        assert json.loads(result.stdout) == [{"name": "report.csv", "size": 8}]
    else:
        assert result.returncode != 0
        assert "Only regular" in result.stderr
