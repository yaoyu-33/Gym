# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from time import time
from uuid import uuid4

from pydantic import JsonValue

from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming


def trajectory_response(
    result: dict[str, JsonValue],
    body: NeMoGymResponseCreateParamsNonStreaming,
    model: str,
    error_type: str | None = None,
) -> NeMoGymResponse:
    """Preserve Hermes's reported messages, stop status, and token usage."""
    output = []
    for message in (result.get("messages") or [])[result.get("n_input", 0) :]:
        role = message.get("role")
        content = message.get("content") or ""
        if isinstance(content, list):
            content = "\n".join(p.get("text", "") for p in content)
        if role == "assistant":
            reasoning = message.get("reasoning") or message.get("reasoning_content")
            if reasoning:
                output.append(
                    {
                        "type": "reasoning",
                        "id": f"rs_{uuid4().hex}",
                        "summary": [{"type": "summary_text", "text": reasoning}],
                    }
                )
            if content:
                output.append(
                    {
                        "type": "message",
                        "id": f"msg_{uuid4().hex}",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": content, "annotations": []}],
                    }
                )
            for call in message.get("tool_calls") or []:
                output.append(
                    {
                        "type": "function_call",
                        "call_id": call["id"],
                        "name": call["function"]["name"],
                        "arguments": call["function"]["arguments"],
                    }
                )
        elif role == "tool":
            output.append({"type": "function_call_output", "call_id": message["tool_call_id"], "output": content})
    failed = bool(error_type or result.get("failed") or result.get("error"))
    completed = bool(result.get("completed")) and not result.get("interrupted")
    usage = result.get("usage") or {}
    inputs, outputs = usage.get("input_tokens", 0), usage.get("output_tokens", 0)
    return NeMoGymResponse.model_validate(
        {
            "id": f"resp_{uuid4().hex}",
            "created_at": int(time()),
            "object": "response",
            "model": model,
            "status": "failed" if failed else "completed" if completed else "incomplete",
            "error": {"code": "server_error", "message": str(result.get("error") or error_type)} if failed else None,
            "metadata": {
                "budget_exhausted": str(bool(result.get("budget_exhausted")) and not failed).lower(),
                "stop_reason": result.get("stop_reason", ""),
            },
            "output": output,
            "tool_choice": body.tool_choice,
            "tools": body.tools,
            "parallel_tool_calls": body.parallel_tool_calls,
            "usage": {
                "input_tokens": inputs,
                "output_tokens": outputs,
                "total_tokens": inputs + outputs,
                "input_tokens_details": {"cached_tokens": usage.get("cached_tokens", 0)},
                "output_tokens_details": {"reasoning_tokens": usage.get("reasoning_tokens", 0)},
            },
        }
    )
