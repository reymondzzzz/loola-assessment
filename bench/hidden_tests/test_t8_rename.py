import ast
from pathlib import Path

from inventory.models import Item


def test_no_qty_attribute_left():
    for path in Path("inventory").glob("*.py"):
        tree = ast.parse(path.read_text())
        attributes = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
        assert "qty" not in attributes, path
    assert not hasattr(Item("pen", 1.0, 2), "qty")


def test_csv_header_unchanged():
    assert Path("data/items.csv").read_text(encoding="utf-8-sig").splitlines()[0].replace(" ", "") == "name,price,qty"


def test_legacy_rows_keep_qty_key():
    from inventory.legacy_report import format_row
    assert "7" in format_row({"name": "pen", "qty": 7, "price": 1.0})
