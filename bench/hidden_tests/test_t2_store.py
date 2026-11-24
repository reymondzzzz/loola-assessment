import pytest

from inventory.models import Item
from inventory.store import Store


def stock(item):
    return getattr(item, "quantity", None) if hasattr(item, "quantity") else item.qty


def test_remove_exact_drops_item():
    store = Store()
    store.add(Item("pen", 1.0, 3))
    store.remove("pen", 3)
    assert "pen" not in store.items


def test_remove_errors():
    store = Store()
    store.add(Item("pen", 1.0, 3))
    with pytest.raises(KeyError):
        store.remove("ink", 1)
    for qty in (0, -2):
        with pytest.raises(ValueError):
            store.remove("pen", qty)
    assert stock(store.items["pen"]) == 3


def test_add_does_not_mutate_argument():
    store = Store()
    first = Item("pen", 1.0, 3)
    store.add(first)
    store.add(Item("pen", 1.0, 2))
    assert stock(first) == 3
    assert stock(store.items["pen"]) == 5


def test_low_stock_inclusive_and_totals():
    store = Store()
    assert store.total_value() == 0.0
    store.add(Item("a", 1.25, 4))
    store.add(Item("b", 3.0, 5))
    assert store.low_stock(5) == ["a", "b"]
    assert store.total_value() == 20.0
