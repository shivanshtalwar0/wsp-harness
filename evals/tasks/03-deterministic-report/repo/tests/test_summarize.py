from reports import summarize

ROWS = [
    {"team": "alpha", "score": 10},
    {"team": "beta", "score": 9},
    {"team": "gamma", "score": 30},
]


def test_highest_first_numerically():
    assert [r["team"] for r in summarize(ROWS)] == ["gamma", "alpha", "beta"]


def test_totals_are_correct():
    assert summarize(ROWS) == [
        {"team": "gamma", "total": 30, "n": 1},
        {"team": "alpha", "total": 10, "n": 1},
        {"team": "beta", "total": 9, "n": 1},
    ]
