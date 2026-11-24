import pytest

from inventory.pricing import apply_discount, bulk_price


def test_discount_bounds():
    assert apply_discount(80.0, 0) == 80.0
    assert apply_discount(80.0, 100) == 0.0
    for percent in (-1, 101):
        with pytest.raises(ValueError):
            apply_discount(10.0, percent)


def test_negative_price_rejected():
    with pytest.raises(ValueError):
        apply_discount(-5.0, 10)


def test_bulk_thresholds_inclusive():
    assert bulk_price(2.0, 9) == 18.0
    assert bulk_price(2.0, 10) == 18.0
    assert bulk_price(2.0, 49) == 88.2
    assert bulk_price(2.0, 50) == 80.0
