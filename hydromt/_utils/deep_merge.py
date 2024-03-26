from typing import Any, Dict


def deep_merge(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, Any]:
    merged = {}

    for k_left, v_left in left.items():
        merged[k_left] = v_left

    for k_right, v_right in right.items():
        if k_right in merged:
            v_left = merged[k_right]
            if isinstance(v_left, dict) and isinstance(v_right, dict):
                merged[k_right] = deep_merge(v_left, v_right)
            else:
                merged[k_right] = v_right
        else:
            merged[k_right] = v_right

    return merged
