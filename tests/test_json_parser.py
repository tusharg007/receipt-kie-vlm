from receipt_kie.metrics import extract_json

EXPECTED = {
    "company": "ACME",
    "address": "1 Road",
    "date": "01/02/2024",
    "total": "12.30",
}


def test_direct_json() -> None:
    parsed = extract_json(
        '{"company":"ACME","address":"1 Road","date":"01/02/2024","total":"12.30"}'
    )
    assert parsed.valid
    assert parsed.value == EXPECTED
    assert parsed.method == "direct"


def test_fenced_and_embedded_json() -> None:
    fenced = extract_json(
        '```json\n{"company":"ACME","address":"1 Road","date":"01/02/2024","total":"12.30"}\n```'
    )
    embedded = extract_json(
        'Result: {"company":"ACME","address":"1 Road","date":"01/02/2024","total":"12.30"} done'
    )
    assert fenced.value == EXPECTED
    assert embedded.value == EXPECTED


def test_controlled_repairs_and_invalid_output() -> None:
    repaired = extract_json(
        "{'company':'ACME','address':'1 Road','date':'01/02/2024','total':'12.30'}"
    )
    invalid = extract_json("I cannot read this receipt.")
    assert repaired.valid
    assert repaired.value == EXPECTED
    assert not invalid.valid
    assert invalid.value is None
