from __future__ import annotations

from receipt_kie.test_usage import hash_id_list


def test_hash_id_list_is_order_sensitive_and_stable() -> None:
    expected = "b64e3448a83a5b86466465080361c1a7e1157a27ddccd4b68069cb18caffb74a"
    assert hash_id_list(["A", "B"]) == expected
    assert hash_id_list(["B", "A"]) != expected
