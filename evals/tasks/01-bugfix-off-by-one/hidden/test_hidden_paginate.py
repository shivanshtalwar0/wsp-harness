from textkit import paginate


def test_last_item_appears():
    items = list(range(10))
    assert paginate(items, 3, 3) == [9]


def test_pages_cover_everything():
    items = list(range(10))
    seen = []
    for page in range(4):
        seen += paginate(items, page, 3)
    assert seen == items


def test_empty_list():
    assert paginate([], 0, 5) == []


def test_page_out_of_range_is_empty():
    assert paginate([1, 2, 3], 9, 3) == []


def test_default_page_size():
    assert len(paginate(list(range(25)), 0)) == 10
    assert len(paginate(list(range(25)), 2)) == 5
