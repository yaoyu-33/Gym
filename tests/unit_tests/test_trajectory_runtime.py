# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pydantic import ValidationError

from nemo_gym.trajectory_runtime import Trajectory


def test_trajectory_rejects_misaligned_training_fields():
    with pytest.raises(ValidationError, match="loss_mask must align"):
        Trajectory(input_ids=[1, 2], loss_mask=[1], logprobs=[0.0, -0.1], reward=1.0)


def test_responses_rollout_projects_multi_turn_training_tokens():
    trajectory = Trajectory.from_responses(
        response={
            "output": [
                {
                    "prompt_token_ids": [10, 11],
                    "generation_token_ids": [12],
                    "generation_log_probs": [-0.1],
                },
                {
                    "prompt_token_ids": [10, 11, 12, 13],
                    "generation_token_ids": [14, 15],
                    "generation_log_probs": [-0.2, -0.3],
                },
            ]
        },
        reward=1.0,
    )

    assert trajectory.input_ids == [10, 11, 12, 13, 14, 15]
    assert trajectory.loss_mask == [0, 0, 1, 0, 1, 1]
    assert trajectory.logprobs == [0.0, 0.0, -0.1, 0.0, -0.2, -0.3]
