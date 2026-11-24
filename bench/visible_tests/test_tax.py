from inventory.tax import compute_tax


def test_tax_rates():
    assert compute_tax(100.0, "DE") == 19.0
    assert compute_tax(100.0, "GB") == 20.0
    assert compute_tax(100.0, "US") == 0.0
