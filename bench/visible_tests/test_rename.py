from inventory.models import Item


def test_item_uses_quantity():
    item = Item("pen", 2.0, quantity=3)
    assert item.quantity == 3
    assert item.total_value() == 6.0
