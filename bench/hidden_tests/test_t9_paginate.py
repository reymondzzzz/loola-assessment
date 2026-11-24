import pytest

from inventory.legacy_report import paginate


def test_paginate_edges():
    assert paginate([], 3) == []
    assert paginate([1], 5) == [[1]]
    assert paginate([1, 2, 3], 1) == [[1], [2], [3]]
    assert paginate([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]
    assert paginate([1, 2, 3, 4], 2) == [[1, 2], [3, 4]]
    with pytest.raises(ValueError):
        paginate([1], 0)
