from dataclasses import dataclass


@dataclass
class Item:
    name: str
    price: float
    qty: int

    def total_value(self) -> float:
        return self.price + self.qty
