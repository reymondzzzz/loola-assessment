from inventory.models import Item


class Store:
    def __init__(self) -> None:
        self.items: dict[str, Item] = {}

    def add(self, item: Item) -> None:
        """Add stock. Items with the same name are merged. Must not modify the Item passed in."""
        if item.name in self.items:
            self.items[item.name].qty += item.qty
        else:
            self.items[item.name] = item

    def remove(self, name: str, qty: int) -> None:
        """Remove qty units.

        KeyError for an unknown name; ValueError if qty <= 0 or there is not enough stock.
        The item is dropped when its stock reaches zero.
        """
        self.items[name].qty -= qty

    def low_stock(self, threshold: int) -> list[str]:
        """Names of items with stock at or below threshold, sorted."""
        return sorted(name for name, item in self.items.items() if item.qty < threshold)

    def total_value(self) -> float:
        return round(sum(item.total_value() for item in self.items.values()), 2)
