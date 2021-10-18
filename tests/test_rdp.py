import numpy as np
import pytest
from src.rdppy import filter as rdp

test_cases = [
    {
        "input": [],
        "threshold": 0.0,
        "output": [],
    },
    {
        "input": [(0, 0)],
        "threshold": 10.0,
        "output": [True],
    },
    {
        "input": [(0, 0), (4, 0)],
        "threshold": 10.0,
        "output": [True, True],
    },
    {
        "input": [(0, 0), (4, 0), (3, 1)],
        "threshold": 10.0,
        "output": [True, False, True],
    },
    {
        "input": [(0, 0), (4, 0), (3, 1)],
        "threshold": 1.4,
        "output": [True, True, True],
    },
    {
        "input": [(0, 0), (4, 0), (3, 1)],
        "threshold": 1.5,
        "output": [True, False, True],
    },
    {
        "input": [
            (0, 0),
            (4, 0),
            (0, 1),
            (1, 1),
            (1, 2),
            (2, 2),
            (2, 3),
            (3, 3),
            (3, 4),
            (5, 4),
        ],
        "threshold": 0.3,
        "output": [True, True, True, True, True, True, True, True, True, True],
    },
    {
        "input": [
            (0, 0),
            (4, 0),
            (0, 1),
            (1, 1),
            (1, 2),
            (2, 2),
            (2, 3),
            (3, 3),
            (3, 4),
            (5, 4),
        ],
        "threshold": 0.9,
        "output": [True, True, True, False, False, False, False, False, True, True],
    },
    {
        "input": [
            (0, 0),
            (4, 0),
            (0, 1),
            (1, 1),
            (1, 2),
            (2, 2),
            (2, 3),
            (3, 3),
            (3, 4),
            (5, 4),
        ],
        "threshold": 1.1,
        "output": [True, True, True, False, False, False, False, False, False, True],
    },
]


@pytest.mark.parametrize(
    "input,threshold,expected",
    [(d["input"], d["threshold"], d["output"]) for d in test_cases],
)
def test_rdp(input, threshold, expected):
    input = np.array(input)
    expected = np.array(expected)
    assert np.all(rdp(input, threshold) == expected)
