# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Framework-neutral primitives for streaming training trajectories.

This module is intentionally independent of any RL algorithm or training
framework. A loaded Gym environment supplies the ``TrajectoryExecutor`` that
owns its agent/environment/evaluator loop; the caller supplies a ``ModelClient``
whose lifecycle remains owned by the RL framework.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Mapping, Sequence
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator


Messages = Sequence[Mapping[str, Any]]
SamplingParams = Mapping[str, Any]
TRAJECTORY_SCHEMA_VERSION = 1


class ModelOutput(BaseModel):
    """One model turn, including exact server-side training token data."""

    model_config = ConfigDict(extra="forbid")

    message: dict[str, Any]
    prompt_token_ids: list[int] | None = None
    generation_token_ids: list[int] | None = None
    generation_logprobs: list[float] | None = None
    raw_response: dict[str, Any] | None = None

    @model_validator(mode="after")
    def validate_generation_arrays(self) -> "ModelOutput":
        if self.generation_logprobs is not None:
            if self.generation_token_ids is None:
                raise ValueError("generation_token_ids are required when generation_logprobs are present")
            if len(self.generation_logprobs) != len(self.generation_token_ids):
                raise ValueError("generation_token_ids and generation_logprobs must have equal lengths")
        return self


class ModelClient(Protocol):
    """Inference boundary implemented by an RL framework or endpoint adapter."""

    async def generate(
        self,
        messages: Messages,
        sampling_params: SamplingParams | None = None,
    ) -> ModelOutput: ...


class OpenAIModelClient:
    """Model client for an OpenAI-compatible Chat Completions endpoint.

    Standard Chat Completions fields are sufficient for semantic trajectories.
    Training-ready trajectories additionally require the endpoint to return
    ``prompt_token_ids``, ``generation_token_ids``, and
    ``generation_log_probs`` on ``choices[0].message``. NeMo-RL's exposed vLLM
    endpoint implements this extension. Stock vLLM can provide the same exact
    data when ``vllm_token_ids=True``: generation IDs come from token-ID
    logprobs and prompt IDs come from its server-side ``/tokenize`` endpoint.
    Both modes avoid local re-tokenization and training/inference token drift.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        *,
        vllm_token_ids: bool = False,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key = api_key
        self._vllm_token_ids = vllm_token_ids

    @property
    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}

    async def _vllm_prompt_token_ids(self, messages: Messages) -> list[int]:
        # Keep the HTTP/server stack optional until the concrete endpoint client is used.
        from nemo_gym.server_utils import get_response_json, raise_for_status, request

        api_root = self._base_url.removesuffix("/v1")
        response = await request(
            method="POST",
            url=f"{api_root}/tokenize",
            json={"model": self._model, "messages": [dict(message) for message in messages]},
            headers=self._headers,
        )
        await raise_for_status(response)
        response_body = await get_response_json(response)
        tokens = response_body.get("tokens") if isinstance(response_body, dict) else None
        if not isinstance(tokens, list) or any(not isinstance(token, int) for token in tokens):
            raise ValueError("vLLM /tokenize response has no integer tokens array")
        return tokens

    @staticmethod
    def _vllm_generation_data(choice: Mapping[str, Any]) -> tuple[list[int], list[float]]:
        logprob_items = (choice.get("logprobs") or {}).get("content") or []
        token_ids: list[int] = []
        logprobs: list[float] = []
        for item in logprob_items:
            token = item.get("token") if isinstance(item, dict) else None
            if not isinstance(token, str) or not token.startswith("token_id:"):
                raise ValueError("vLLM did not return token-ID logprobs")
            token_ids.append(int(token.removeprefix("token_id:")))
            logprobs.append(float(item["logprob"]))
        return token_ids, logprobs

    async def generate(
        self,
        messages: Messages,
        sampling_params: SamplingParams | None = None,
    ) -> ModelOutput:
        # Keep the HTTP/server stack optional until the concrete endpoint client is used.
        from nemo_gym.server_utils import get_response_json, raise_for_status, request

        sampling = dict(sampling_params or {})
        reserved = {"messages", "model", "stream"}.intersection(sampling)
        if reserved:
            raise ValueError(f"sampling_params may not override: {', '.join(sorted(reserved))}")
        payload = {
            "model": self._model,
            "messages": [dict(message) for message in messages],
            "stream": False,
            "logprobs": True,
            **sampling,
        }
        if self._vllm_token_ids:
            payload["return_tokens_as_token_ids"] = True
        response = await request(
            method="POST",
            url=f"{self._base_url}/chat/completions",
            json=payload,
            headers=self._headers,
        )
        await raise_for_status(response)
        response_body = await get_response_json(response)
        try:
            choice = response_body["choices"][0]
            message = dict(choice["message"])
        except (IndexError, KeyError, TypeError) as exc:
            raise ValueError("Chat Completions response has no choices[0].message") from exc
        prompt_token_ids = message.pop("prompt_token_ids", None)
        generation_token_ids = message.pop("generation_token_ids", None)
        generation_logprobs = message.pop("generation_log_probs", None)
        if self._vllm_token_ids and prompt_token_ids is None:
            prompt_token_ids = await self._vllm_prompt_token_ids(messages)
        if self._vllm_token_ids and generation_token_ids is None:
            generation_token_ids, generation_logprobs = self._vllm_generation_data(choice)
        return ModelOutput(
            message=message,
            prompt_token_ids=prompt_token_ids,
            generation_token_ids=generation_token_ids,
            generation_logprobs=generation_logprobs,
            raw_response=response_body,
        )


class Trajectory(BaseModel):
    """Framework-neutral result of one task/sample rollout.

    Token-aligned fields either all use the length of ``input_ids`` or are
    absent. Logprobs at masked prompt/environment positions are conventionally
    zero; ``loss_mask`` is authoritative.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: int = TRAJECTORY_SCHEMA_VERSION
    task_id: str
    sample_id: str
    input_ids: list[int] | None = None
    loss_mask: list[int] | None = None
    logprobs: list[float] | None = None
    reward: float | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    messages: list[dict[str, Any]] = Field(default_factory=list)
    steps: list[dict[str, Any]] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_training_arrays(self) -> "Trajectory":
        if self.schema_version != TRAJECTORY_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported trajectory schema_version {self.schema_version}; expected {TRAJECTORY_SCHEMA_VERSION}"
            )
        if self.input_ids is None:
            if self.loss_mask is not None or self.logprobs is not None:
                raise ValueError("loss_mask and logprobs require input_ids")
            return self
        if self.loss_mask is None or len(self.loss_mask) != len(self.input_ids):
            raise ValueError("loss_mask must be present and aligned with input_ids")
        if any(value not in (0, 1) for value in self.loss_mask):
            raise ValueError("loss_mask values must be 0 or 1")
        if self.logprobs is not None and len(self.logprobs) != len(self.input_ids):
            raise ValueError("logprobs must be aligned with input_ids")
        return self

    @classmethod
    def from_responses(
        cls,
        *,
        task_id: str,
        sample_id: str,
        messages: Messages,
        response: Mapping[str, Any],
        reward: float | None = None,
        metrics: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "Trajectory":
        """Project a token-bearing Responses rollout into one training trajectory.

        Each generated output item carries the cumulative prompt seen by that
        model call. This projection retains only the newly appended prompt tokens
        between calls, marks them non-trainable, and marks the sampled generation
        trainable. It therefore supports multi-turn and tool-interleaved rollouts
        without re-tokenizing any content.
        """
        input_ids: list[int] = []
        loss_mask: list[int] = []
        logprobs: list[float] = []
        seen_token_ids: list[int] = []
        steps: list[dict[str, Any]] = []

        for raw_item in response.get("output", []):
            if not isinstance(raw_item, Mapping):
                continue
            item = dict(raw_item)
            token_fields = (
                item.get("prompt_token_ids"),
                item.get("generation_token_ids"),
                item.get("generation_log_probs"),
            )
            if all(value is None for value in token_fields):
                continue
            if not all(isinstance(value, list) for value in token_fields):
                raise ValueError("Responses output item has partial token metadata")
            prompt_token_ids, generation_token_ids, generation_logprobs = token_fields
            if any(not isinstance(token_id, int) for token_id in prompt_token_ids):
                raise ValueError("prompt_token_ids must contain only integers")
            if any(not isinstance(token_id, int) for token_id in generation_token_ids):
                raise ValueError("generation_token_ids must contain only integers")
            if len(generation_token_ids) != len(generation_logprobs):
                raise ValueError("generation_token_ids and generation_log_probs must align")
            if seen_token_ids != prompt_token_ids[: len(seen_token_ids)]:
                raise ValueError("Responses token metadata is not prefix-contiguous")

            new_prompt_token_ids = prompt_token_ids[len(seen_token_ids) :]
            input_ids.extend(new_prompt_token_ids)
            loss_mask.extend([0] * len(new_prompt_token_ids))
            logprobs.extend([0.0] * len(new_prompt_token_ids))
            generation_start = len(input_ids)
            input_ids.extend(generation_token_ids)
            loss_mask.extend([1] * len(generation_token_ids))
            logprobs.extend(float(value) for value in generation_logprobs)
            generation_end = len(input_ids)

            semantic_item = {
                key: value
                for key, value in item.items()
                if key
                not in {
                    "prompt_token_ids",
                    "generation_token_ids",
                    "generation_log_probs",
                }
            }
            steps.append(
                {
                    "type": "model",
                    "output": semantic_item,
                    "generation_span": [generation_start, generation_end],
                }
            )
            seen_token_ids = list(prompt_token_ids) + list(generation_token_ids)

        semantic_messages = [dict(message) for message in messages]
        semantic_messages.extend(dict(item) for item in response.get("output", []) if isinstance(item, Mapping))
        has_training_tokens = any(loss_mask)
        return cls(
            task_id=task_id,
            sample_id=sample_id,
            input_ids=input_ids if has_training_tokens else None,
            loss_mask=loss_mask if has_training_tokens else None,
            logprobs=logprobs if has_training_tokens else None,
            reward=reward,
            metrics=dict(metrics or {}),
            metadata=dict(metadata or {}),
            messages=semantic_messages,
            steps=steps,
        )

    @classmethod
    def from_model_output(
        cls,
        *,
        task_id: str,
        sample_id: str,
        messages: Messages,
        output: ModelOutput,
        reward: float | None = None,
        metrics: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "Trajectory":
        """Build a one-turn trajectory without re-tokenizing model output."""
        input_ids = None
        loss_mask = None
        logprobs = None
        if output.prompt_token_ids is not None and output.generation_token_ids is not None:
            input_ids = output.prompt_token_ids + output.generation_token_ids
            loss_mask = [0] * len(output.prompt_token_ids) + [1] * len(output.generation_token_ids)
            if output.generation_logprobs is not None:
                logprobs = [0.0] * len(output.prompt_token_ids) + output.generation_logprobs
        semantic_messages = [dict(message) for message in messages]
        semantic_messages.append(output.message)
        return cls(
            task_id=task_id,
            sample_id=sample_id,
            input_ids=input_ids,
            loss_mask=loss_mask,
            logprobs=logprobs,
            reward=reward,
            metrics=dict(metrics or {}),
            metadata=dict(metadata or {}),
            messages=semantic_messages,
            steps=[{"type": "model", "output": output.message}],
        )


TrajectoryExecutor = Callable[
    [Mapping[str, Any], ModelClient, str, SamplingParams | None],
    Awaitable[Trajectory],
]


class TrajectoryRunner:
    """Run a Gym-owned loop and stream trajectories in completion order."""

    def __init__(self, executor: TrajectoryExecutor, *, max_concurrency: int | None = None) -> None:
        if max_concurrency is not None and max_concurrency < 1:
            raise ValueError("max_concurrency must be positive")
        self._executor = executor
        self._semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency is not None else None

    async def _run_one(
        self,
        task: Mapping[str, Any],
        model: ModelClient,
        sample_id: str,
        sampling: SamplingParams | None,
    ) -> Trajectory:
        if self._semaphore is None:
            return await self._executor(task, model, sample_id, sampling)
        async with self._semaphore:
            return await self._executor(task, model, sample_id, sampling)

    async def run(
        self,
        tasks: Iterable[Mapping[str, Any]],
        *,
        model: ModelClient,
        n: int = 1,
        sampling: SamplingParams | None = None,
    ) -> AsyncIterator[Trajectory]:
        """Yield task/sample rollouts as each finishes."""
        if n < 1:
            raise ValueError("n must be positive")
        pending = []
        for task_index, task in enumerate(tasks):
            task_id = str(task.get("task_id", task_index))
            for sample_index in range(n):
                sample_id = f"{task_id}:{sample_index}"
                pending.append(asyncio.create_task(self._run_one(task, model, sample_id, sampling)))
        try:
            for future in asyncio.as_completed(pending):
                yield await future
        finally:
            for task in pending:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
