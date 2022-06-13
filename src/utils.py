"""Module for functions that are reused across the src scripts."""

import numpy as np


def get_hexcolor_by_value(value: float, all_values: list, reverse: bool = False) -> str:
    """Returns a hexcolor code based on a provided and its relation to all values in a list. If reverse=False, smaller
    values are red and higher values are green and vice versa."""
    mean = np.mean(all_values)
    std = np.std(all_values)
    min_val = max(min(all_values), mean - std)
    max_val = min(max(all_values), mean + std)
    adj_mean = ((min_val+max_val)/2)
    if value <= adj_mean:
        r = 170
        g = int(255*max(0.0, value-(adj_mean-value))/adj_mean)
        b = 0
    else:
        r = int(255*max(0.0, max_val-value)/adj_mean)
        g = 170
        b = 0
    if reverse:
        r, g = g, r
    return "#%s%s%s" % tuple([hex(c)[2:].rjust(2, "0") for c in (r, g, b)])