"""Invoice maths."""


def invoice_total(items, tax):
    """Total for ``items`` (dicts with an ``amount``) including ``tax``."""
    subtotal = sum(item["amount"] for item in items)
    return round(subtotal * (1 + tax), 2)
