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
"""Token-aligned training trajectory returned by an agent run."""

from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, model_validator


class Trajectory(BaseModel):
    """Minimal framework-neutral result consumed by an RL trainer."""

    model_config = ConfigDict(extra="forbid")

    input_ids: list[int]
    loss_mask: list[int]
    logprobs: list[float]
    reward: float

    @model_validator(mode="after")
    def validate_training_arrays(self) -> "Trajectory":
        if not self.input_ids:
            raise ValueError("input_ids must not be empty")
        if len(self.loss_mask) != len(self.input_ids):
            raise ValueError("loss_mask must align with input_ids")
        if len(self.logprobs) != len(self.input_ids):
            raise ValueError("logprobs must align with input_ids")
        if any(value not in (0, 1) for value in self.loss_mask):
            raise ValueError("loss_mask values must be 0 or 1")
        return self

    @classmethod
    def from_responses(
        cls,
        *,
        response: Mapping[str, Any],
        reward: float,
    ) -> "Trajectory":
        """Flatten exact multi-turn Responses tokens without re-tokenizing."""
        input_ids: list[int] = []
        loss_mask: list[int] = []
        logprobs: list[float] = []
        seen_token_ids: list[int] = []

        for item in response.get("output", []):
            if not isinstance(item, Mapping) or not item.get("generation_token_ids"):
                continue
            prompt_token_ids = item["prompt_token_ids"]
            generation_token_ids = item["generation_token_ids"]
            generation_logprobs = item["generation_log_probs"]
            if seen_token_ids != prompt_token_ids[: len(seen_token_ids)]:
                raise ValueError("Responses token metadata is not prefix-contiguous")
            if len(generation_token_ids) != len(generation_logprobs):
                raise ValueError("generation token IDs and logprobs must align")

            new_prompt_token_ids = prompt_token_ids[len(seen_token_ids) :]
            input_ids.extend(new_prompt_token_ids)
            loss_mask.extend([0] * len(new_prompt_token_ids))
            logprobs.extend([0.0] * len(new_prompt_token_ids))
            input_ids.extend(generation_token_ids)
            loss_mask.extend([1] * len(generation_token_ids))
            logprobs.extend(float(value) for value in generation_logprobs)
            seen_token_ids = list(prompt_token_ids) + list(generation_token_ids)

        return cls(
            input_ids=input_ids,
            loss_mask=loss_mask,
            logprobs=logprobs,
            reward=reward,
        )
