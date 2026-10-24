from __future__ import annotations

import json

import httpx
import pytest

from receipt_kie.bolt import BoltReviewAgent


def _row():
    return {
        "raw_output": json.dumps(
            {"company": "ACME", "address": "1 Road", "date": "01/02/2024", "total": "12.50"}
        )
    }


def _tool(name="inspect_receipt", arguments="{}"):
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": "call1", "type": "function", "function": {"name": name, "arguments": arguments}}
        ],
    }


def test_agent_executes_only_diagnostics_then_a_bounded_decision() -> None:
    calls = []

    def respond(request):
        payload = json.loads(request.content)
        calls.append(payload)
        if len(calls) == 1:
            message = _tool()
        else:
            message = {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "action": "request_review",
                        "reason": "insufficient_evidence",
                        "fields": ["total"],
                    }
                ),
            }
        return httpx.Response(
            200, json={"choices": [{"message": message, "finish_reason": "stop"}]}
        )

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        result = BoltReviewAgent(client=client).review(_row())
    assert len(calls) == 2
    assert calls[1]["messages"][-2]["role"] == "tool"
    assert calls[1]["response_format"]["json_schema"]["schema"]["additionalProperties"] is False
    assert calls[1]["reasoning_effort"] == "none"
    assert result["status"] == "completed" and not result["image_verified"]
    assert result["trace"][0]["tool"] == "inspect_receipt"
    assert all("image_url" not in json.dumps(call) for call in calls)


@pytest.mark.parametrize("tool", [_tool("run_shell"), _tool(arguments='{"command":"delete"}')])
def test_agent_rejects_unlisted_tools_and_arguments(tool) -> None:
    def respond(request):
        return httpx.Response(200, json={"choices": [{"message": tool}]})

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        result = BoltReviewAgent(client=client).review(_row())
    assert result["action"] == "abstain"
    assert result["status"] == "unavailable_or_invalid"
    assert not result["trace"]


def test_agent_never_accepts_an_approval_or_field_replacement() -> None:
    calls = []

    def respond(request):
        calls.append(request)
        message = (
            _tool()
            if len(calls) == 1
            else {
                "content": json.dumps(
                    {
                        "action": "approve",
                        "reason": "insufficient_evidence",
                        "fields": [],
                        "corrected_total": "999.00",
                    }
                )
            }
        )
        return httpx.Response(200, json={"choices": [{"message": message}]})

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        assert BoltReviewAgent(client=client).review(_row())["action"] == "abstain"


def test_upstream_failure_does_not_turn_into_approval() -> None:
    with httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(503))) as client:
        assert BoltReviewAgent(client=client).review(_row())["action"] == "abstain"
