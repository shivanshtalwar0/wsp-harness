from textkit import paginate


def test_first_page_is_full():
    assert paginate(list(range(10)), 0, 3) == [0, 1, 2]


def test_second_page_is_full():
    assert paginate(list(range(10)), 1, 3) == [3, 4, 5]
