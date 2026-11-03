from reports import summarize

TIES = [
    {"team": "zeta", "score": 5},
    {"team": "alpha", "score": 5},
    {"team": "mike", "score": 5},
    {"team": "big", "score": 100},
    {"team": "small", "score": 2},
]


def test_ties_break_by_name_ascending():
    assert [r["team"] for r in summarize(TIES)] == ["big", "alpha", "mike", "zeta", "small"]


def test_numeric_not_lexicographic():
    rows = [{"team": "a", "score": 9}, {"team": "b", "score": 10}]
    assert [r["team"] for r in summarize(rows)] == ["b", "a"]


def test_repeated_calls_identical():
    assert summarize(TIES) == summarize(list(reversed(TIES)))


def test_empty():
    assert summarize([]) == []


def test_aggregates_multiple_rows_per_team():
    rows = [{"team": "a", "score": 4}, {"team": "a", "score": 6}, {"team": "b", "score": 9}]
    assert summarize(rows) == [{"team": "a", "total": 10, "n": 2}, {"team": "b", "total": 9, "n": 1}]
