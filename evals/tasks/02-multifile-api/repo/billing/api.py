from .calc import invoice_total


def create_invoice(items, region="US"):
    """Build a simple invoice dict for a region."""
    tax = 0.2 if region == "UK" else 0.0
    return {"items": items, "region": region, "total": invoice_total(items, tax)}
