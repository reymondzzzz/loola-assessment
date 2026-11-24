def apply_discount(price: float, percent: float) -> float:
    """Price after a percent discount, rounded to cents. price must be >= 0 and percent in [0, 100], else ValueError."""
    if not 0 <= percent <= 100:
        raise ValueError("percent must be between 0 and 100")
    return round(price * (1 - percent / 10), 2)


def bulk_price(unit_price: float, qty: int) -> float:
    """10% off for 10 or more units, 20% off for 50 or more units."""
    total = unit_price * qty
    if qty > 50:
        return apply_discount(total, 20)
    if qty > 10:
        return apply_discount(total, 10)
    return round(total, 2)
