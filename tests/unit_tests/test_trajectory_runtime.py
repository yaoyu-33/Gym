# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio

import pytest
from pydantic import ValidationError

from nemo_gym.trajectory_runtime import ModelOutput, Trajectory, TrajectoryRunner


class FakeModelClient:
    async def generate(self, messages, sampling_params=None):
        delay = float(messages[0]["content"])
        await asyncio.sleep(delay)
        return ModelOutput(
            message={"role": "assistant", "content": "4"},
            prompt_token_ids=[10, 11],
            generation_token_ids=[12],
            generation_logprobs=[-0.25],
        )


async def exact_match_executor(task, model, sample_id, sampling):
    output = await model.generate(task["messages"], sampling)
    reward = float(output.message["content"] == task["expected_answer"])
    return Trajectory.from_model_output(
        task_id=task["task_id"],
        sample_id=sample_id,
        messages=task["messages"],
        output=output,
        reward=reward,
    )


@pytest.mark.asyncio
async def test_runner_streams_training_ready_trajectories_as_completed():
    runner = TrajectoryRunner(exact_match_executor)
    tasks = [
        {"task_id": "slow", "messages": [{"role": "user", "content": "0.02"}], "expected_answer": "4"},
        {"task_id": "fast", "messages": [{"role": "user", "content": "0"}], "expected_answer": "4"},
    ]

    trajectories = [trajectory async for trajectory in runner.run(tasks, model=FakeModelClient(), n=2)]

    assert trajectories[0].task_id == "fast"
    assert {trajectory.sample_id for trajectory in trajectories} == {"slow:0", "slow:1", "fast:0", "fast:1"}
    assert all(trajectory.input_ids == [10, 11, 12] for trajectory in trajectories)
    assert all(trajectory.loss_mask == [0, 0, 1] for trajectory in trajectories)
    assert all(trajectory.logprobs == [0.0, 0.0, -0.25] for trajectory in trajectories)
    assert all(trajectory.reward == 1.0 for trajectory in trajectories)


def test_trajectory_rejects_misaligned_training_fields():
    with pytest.raises(ValidationError, match="loss_mask must be present and aligned"):
        Trajectory(task_id="task", sample_id="task:0", input_ids=[1, 2], loss_mask=[1])
