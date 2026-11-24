from inventory.pricing import apply_discount, bulk_price


def test_apply_discount():
    assert apply_discount(200.0, 25) == 150.0


def test_bulk_price_small_order():
    assert bulk_price(2.0, 5) == 10.0


def test_bulk_price_large_order():
    assert bulk_price(1.0, 100) == 80.0
