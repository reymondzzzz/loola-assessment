from inventory.legacy_report import build_report


def test_report_includes_every_row():
    rows = [{"name": f"item-{i}", "qty": i, "price": 1.0} for i in range(5)]
    report = build_report(rows, page_size=2)
    assert report.count("item-") == 5
