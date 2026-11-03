import pytest

from billing import create_invoice, invoice_total

ITEMS = [{"amount": 100.0}]


def test_keyword_only():
    with pytest.raises(TypeError):
        invoice_total(ITEMS, 0.2)


def test_tax_by_keyword():
    assert invoice_total(ITEMS, tax_rate=0.2) == 120.0


def test_discount_before_tax():
    assert invoice_total(ITEMS, tax_rate=0.2, discount_rate=0.1) == 108.0


def test_caller_updated():
    inv = create_invoice(ITEMS, region="UK")
    assert inv["total"] == 120.0
