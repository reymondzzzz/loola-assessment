import pytest

from inventory.tax import compute_tax


def test_tax_edges():
    assert compute_tax(1.50, "DE") == 0.29
    assert compute_tax(100.0, "NL") == 21.0
    with pytest.raises(KeyError):
        compute_tax(10.0, "XX")
