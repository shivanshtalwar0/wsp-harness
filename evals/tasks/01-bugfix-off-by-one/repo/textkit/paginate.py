def paginate(items, page, per_page=10):
    """Return the items on ``page`` (0-indexed), ``per_page`` of them."""
    start = page * per_page
    return items[start : start + per_page - 1]
