from hydromt._utils.deep_merge import deep_merge


def test_deep_merge_simple():
    left = {"a": 1, "b": 2, "c": 4}
    right = {"d": 1, "b": 24, "e": 4}

    assert deep_merge(left, right) == {"a": 1, "b": 24, "c": 4, "d": 1, "e": 4}


def test_deep_merge_nested_overwrite():
    left = {"a": 1, "b": 2, "c": {"d": {"b": {"e": 4}}}}
    right = {"c": {"d": {"b": {"e": 8}}}}

    assert deep_merge(left, right) == {"a": 1, "b": 2, "c": {"d": {"b": {"e": 8}}}}


def test_deep_merge_disjoint():
    left = {"a": {"b": 2, "c": {"d": {"b": {"e": 4}}}}}
    right = {"q": {"d": {"b": {"e": 8}}}}

    assert deep_merge(left, right) == {
        "q": {"d": {"b": {"e": 8}}},
        "a": {"b": 2, "c": {"d": {"b": {"e": 4}}}},
    }


def test_deep_merge_override_dict_with_value():
    left = {"a": {"b": 2, "c": {"d": {"b": {"e": 4}}}}}
    right = {"a": 3}

    assert deep_merge(left, right) == {"a": 3}
