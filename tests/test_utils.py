"""Testing for the internal hydromt utility functions."""
from hydromt.utils import partition_dictionaries


def test_flat_dict_partition():
    left = {"a": 1, "b": 2, "pi": 3.14}
    right = {"a": 1, "b": 2, "e": 2.71}
    common, left_less_right, right_less_left = partition_dictionaries(left, right)
    assert common == {"a": 1, "b": 2}
    assert left_less_right == {"pi": 3.14}
    assert right_less_left == {"e": 2.71}


def test_nested_disjoint_leaves():
    left = {"a": 1, "b": 2, "maths": {"constants": {"pi": 3.14}}}
    right = {"a": 1, "b": 2, "maths": {"constants": {"e": 2.71}}}
    common, left_less_right, right_less_left = partition_dictionaries(left, right)
    assert common == {"a": 1, "b": 2, "maths": {"constants": {}}}
    assert left_less_right == {"maths": {"constants": {"pi": 3.14}}}
    assert right_less_left == {"maths": {"constants": {"e": 2.71}}}


def test_nested_common_siblings():
    left = {
        "a": 1,
        "b": 2,
        "maths": {
            "constants": {"pi": 3.14},
            "integration": {"numeric": None, "analytic": None},
        },
    }
    right = {
        "a": 1,
        "b": 2,
        "maths": {
            "constants": {"e": 2.71},
            "integration": {"numeric": None, "analytic": None},
        },
    }
    common, left_less_right, right_less_left = partition_dictionaries(left, right)
    assert common == {
        "a": 1,
        "b": 2,
        "maths": {"constants": {}, "integration": {"numeric": None, "analytic": None}},
    }
    assert left_less_right == {"maths": {"constants": {"pi": 3.14}}}
    assert right_less_left == {"maths": {"constants": {"e": 2.71}}}


def test_nested_key_conflict():
    left = {
        "a": 1,
        "b": 2,
        "c": 3,
        "d": 4,
        "e": 5,
        "maths": {"constants": {"pi": 3.14}},
    }
    right = {"a": 1, "b": 2, "c": 3, "d": 4, "maths": {"constants": {"e": 2.71}}}

    common, left_less_right, right_less_left = partition_dictionaries(left, right)

    assert common == {"a": 1, "b": 2, "c": 3, "d": 4, "maths": {"constants": {}}}
    assert left_less_right == {
        "e": 5,
        "maths": {"constants": {"pi": 3.14}},
    }
    assert right_less_left == {
        "maths": {"constants": {"e": 2.71}},
    }


def test_common_ancestory_distinct_children():
    left = {
        "a": {"i": -1, "ii": -2, "iii": -3},
        "b": 2,
        "c": 3,
        "d": 4,
        "e": 5,
        "maths": {"constants": {"pi": 3.14}},
    }
    right = {
        "a": {"i": -1, "ii": -2, "iii": -3},
        "b": 2,
        "c": 3,
        "d": 4,
        "maths": {"constants": {"e": 2.71}},
    }

    common, left_less_right, right_less_left = partition_dictionaries(left, right)
    assert common == {
        "a": {"i": -1, "ii": -2, "iii": -3},
        "b": 2,
        "c": 3,
        "d": 4,
        "maths": {"constants": {}},
    }
    assert left_less_right == {
        "e": 5,
        "maths": {"constants": {"pi": 3.14}},
    }
    assert right_less_left == {
        "maths": {"constants": {"e": 2.71}},
    }
