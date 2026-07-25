import json

from receipt_kie.prompts import CANONICAL_FIELDS, USER_PROMPT, build_messages, canonical_json


def test_canonical_target_order_and_missing_values() -> None:
    rendered = canonical_json({"total": 5, "company": "Shop"})
    assert tuple(json.loads(rendered)) == CANONICAL_FIELDS
    assert json.loads(rendered) == {
        "company": "Shop",
        "address": "",
        "date": "",
        "total": "5",
    }


def test_multimodal_chat_message_structure() -> None:
    messages = build_messages("receipt.jpg", {"company": "Shop"})
    assert [message["role"] for message in messages] == ["system", "user", "assistant"]
    assert messages[1]["content"][0]["type"] == "image"
    assert "valid JSON only" in USER_PROMPT
