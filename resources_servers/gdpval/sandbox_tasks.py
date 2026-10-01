# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Minimal file-task input contract for sandboxed GDP harnesses."""

import argparse
import json
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator
from typing_extensions import Self


WORKDIR = "/workspace"
INPUT_DIR = f"{WORKDIR}/input"
OUTPUT_DIR = f"{WORKDIR}/output"


def relative_file(value: str) -> str:
    """Accept only unambiguous, relative POSIX file paths."""
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or "\\" in value or "\x00" in value:
        raise ValueError(f"Unsafe reference path: {value!r}")
    if path.as_posix() != value or value == ".":
        raise ValueError(f"Noncanonical reference path: {value!r}")
    return value


class GDPFileTask(BaseModel):
    """Validate task inputs without exposing verifier metadata to the harness."""

    model_config = ConfigDict(extra="ignore")
    task_id: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    reference_files: list[str] = Field(default_factory=list)
    reference_file_urls: list[str] = Field(default_factory=list)

    @field_validator("reference_files", "reference_file_urls", mode="before")
    @classmethod
    def parse_lists(cls, value: object) -> object:
        return json.loads(value) if isinstance(value, str) else value

    @model_validator(mode="after")
    def validate_references(self) -> Self:
        if len(self.reference_files) != len(self.reference_file_urls):
            raise ValueError("Every reference file must have one download URL")
        if len(set(self.reference_files)) != len(self.reference_files):
            raise ValueError("Duplicate reference paths")
        for name in self.reference_files:
            relative_file(name)
        for url in self.reference_file_urls:
            parsed = urlsplit(url)
            if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
                raise ValueError("Reference URLs must be HTTPS URLs without credentials")
        return self


def prepare_row(row: dict[str, JsonValue]) -> dict[str, JsonValue]:
    """Adapt a GDP row for single_agent_turn, retaining verifier-only metadata."""
    task = GDPFileTask.model_validate(row)
    references = "\n".join(f"- {INPUT_DIR}/{name}" for name in task.reference_files) or "None"
    prompt = (
        f"Complete the following task using your sandbox tools. Work in {WORKDIR}.\n"
        f"Reference files (do not modify):\n{references}\n\n"
        f"Save all final deliverables directly in {OUTPUT_DIR}, with no subdirectories. "
        "Only files in that directory will be submitted. Keep scripts, logs and scratch files elsewhere. "
        "When finished, reply with a brief summary and the names of your deliverables.\n\n"
        f"Task:\n{task.prompt}"
    )
    result = dict(row)
    params = dict(row.get("responses_create_params") or {})
    params["input"] = [{"role": "user", "content": prompt}]
    result["responses_create_params"] = params
    return result


def main() -> None:
    """Prepare a separate JSONL; never overwrite the source dataset."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [prepare_row(json.loads(line)) for line in args.input.read_text().splitlines() if line.strip()]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
