import pytest

from inventory.models import Item
from inventory.store import Store


def test_item_total_value():
    assert Item("pen", 2.5, 4).total_value() == 10.0


def test_remove_insufficient_stock():
    store = Store()
    store.add(Item("pen", 1.0, 3))
    with pytest.raises(ValueError):
        store.remove("pen", 5)


def test_low_stock():
    store = Store()
    store.add(Item("a", 1.0, 2))
    store.add(Item("b", 1.0, 9))
    assert store.low_stock(4) == ["a"]
