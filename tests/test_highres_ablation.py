from __future__ import annotations

from types import SimpleNamespace

from PIL import Image

from receipt_kie.inference import resize_to_longest_edge
from receipt_kie.metrics import has_repetition_failure
from receipt_kie.model import configure_image_processor


def test_processor_overrides_are_independent_and_defaults_are_preserved() -> None:
    image_processor = SimpleNamespace(
        size={"longest_edge": 2048},
        max_image_size={"longest_edge": 512},
        do_image_splitting=True,
    )
    unchanged = configure_image_processor(image_processor, {})
    assert unchanged == {
        "size": {"longest_edge": 2048},
        "max_image_size": {"longest_edge": 512},
        "do_image_splitting": True,
    }
    configured = configure_image_processor(
        image_processor,
        {
            "image_longest_edge": 1024,
            "max_image_patch_edge": 384,
            "do_image_splitting": False,
        },
    )
    assert configured == {
        "size": {"longest_edge": 1024},
        "max_image_size": {"longest_edge": 384},
        "do_image_splitting": False,
    }


def test_explicit_resize_matches_idefics_even_dimension_rule() -> None:
    image = Image.new("RGB", (4961, 7016), color="white")
    resized = resize_to_longest_edge(image, 512)
    assert resized.size == (362, 512)


def test_repetition_failure_uses_repeated_phrase_not_single_words() -> None:
    repeated = "JALAN SATU DUA " * 3
    normal = '{"company":"ACME","address":"1 Jalan Satu","date":"01/01/2024"}'
    assert has_repetition_failure(repeated)
    assert not has_repetition_failure(normal)
