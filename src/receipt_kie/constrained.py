"""Optional token filtering for the existing Transformers/PEFT model stack."""

from __future__ import annotations

from typing import Any

from receipt_kie.structured import receipt_schema


class ReceiptTokenConstraint:
    """Reuse tokenizer indexing, but create fresh parser state for every receipt."""

    def __init__(self, tokenizer: Any) -> None:
        try:
            from lmformatenforcer.integrations.transformers import (
                build_token_enforcer_tokenizer_data,
            )
        except ImportError as exc:
            raise ImportError(
                "Structured decoding requires: pip install -r requirements-structured.txt"
            ) from exc
        self.tokenizer_data = build_token_enforcer_tokenizer_data(tokenizer)

    def prefix_allowed_tokens_fn(self) -> Any:
        from lmformatenforcer import JsonSchemaParser
        from lmformatenforcer.integrations.transformers import (
            build_transformers_prefix_allowed_tokens_fn,
        )

        return build_transformers_prefix_allowed_tokens_fn(
            self.tokenizer_data, JsonSchemaParser(receipt_schema())
        )
