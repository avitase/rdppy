import numpy as np
import pytest
from src.rdppy import lossy_compress as rdp

test_cases = [
    {
        "input": [(0, 0), (4, 0), (3, 1)],
        "threshold": 1.4,
        "output": [(0, 0), (4, 0), (3, 1)],
    },
    {
        "input": [(0, 0), (4, 0), (3, 1)],
        "threshold": 1.5,
        "output": [(0, 0), (3, 1)],
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
        "output": [
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
        "output": [(0, 0), (4, 0), (0, 1), (3, 4), (5, 4)],
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
        "output": [(0, 0), (4, 0), (0, 1), (5, 4)],
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
