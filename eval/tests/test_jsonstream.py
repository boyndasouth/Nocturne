import json

import pytest

from eval.datasets.jsonstream import iter_json_array

CASES = [
    [],
    [{"a": 1}],
    [{"text": "has ] and , and } inside", "n": [1, 2, {"x": None}]}, {"b": "é ✓"}],
    [12345678, -0.5e10, "s", True, None, [], {}],
    [{"big": "y" * 5000}, {"big": "z" * 5000}],
]


@pytest.mark.parametrize("chunk_size", [1, 3, 7, 64, 1 << 20])
@pytest.mark.parametrize("data", CASES)
def test_matches_json_load_at_any_chunk_size(tmp_path, data, chunk_size):
    path = tmp_path / "a.json"
    path.write_text("﻿ \n" + json.dumps(data, indent=1, ensure_ascii=False) + "\n ", "utf-8")
    assert list(iter_json_array(path, chunk_size=chunk_size)) == data


def test_number_cut_at_a_chunk_edge_is_not_split(tmp_path):
    path = tmp_path / "n.json"
    path.write_text("[1234567, 89]", "utf-8")
    assert list(iter_json_array(path, chunk_size=4)) == [1234567, 89]


@pytest.mark.parametrize("text", ['{"a": 1}', "[1, 2", "[1 2]", '[{"a": ]'])
def test_bad_input_raises(tmp_path, text):
    path = tmp_path / "bad.json"
    path.write_text(text, "utf-8")
    with pytest.raises(ValueError):  # JSONDecodeError is a ValueError
        list(iter_json_array(path, chunk_size=3))


def test_longmemeval_fixture_loads_the_same(tmp_path):
    from pathlib import Path

    from eval.datasets import longmemeval

    fixture = Path(__file__).parent / "fixtures" / "longmemeval_mini.json"
    expected = [longmemeval.parse_instance(x) for x in json.loads(fixture.read_text("utf-8"))]
    assert longmemeval.load(fixture) == expected
