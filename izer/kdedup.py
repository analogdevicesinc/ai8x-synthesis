###################################################################################################
# Copyright (C) 2022-2023 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
###################################################################################################
"""
Deduplicate kernel values
"""
import operator as opr
from functools import reduce
from typing import List, Optional, Tuple

import numpy as np

import xxhash

from . import state
from .eprint import nprint
from .utils import plural


def deduplicate(
        weights_in: List[np.ndarray],
        layers: int,
        quantization: List[int],
        processor_map: List[int],
        kind: str = 'kernels',
) -> Tuple[List[Optional[int]], List[Optional[np.ndarray]]]:
    """
    Remove duplicates from weights and return list of pointers, and the deduplicated weights.
    Compares full weights and their shape (does not try to match a subset).
    """
    weights_out: List[Optional[np.ndarray]] = []
    weight_hash: List[Optional[int]] = []
    weight_ptrs: List[Optional[int]] = []
    h3 = xxhash.xxh3_128()
    n: int = 0
    saved_weight_bytes: int = 0

    for ll, w in enumerate(weights_in):
        if w is None or ll >= layers:  # Empty or past the weights we need
            weight_hash.append(None)
            weight_ptrs.append(None if w is None else ll)
            weights_out.append(w)
            continue

        # In order to deduplicate, the contents must be the same, and the shape.
        # For kernels, quantization and processor_map must match as well.
        # Contents are first checked using a hash; if the hash matches, a full compare is done.
        # There is no checking for subsets (smaller weights part of larger weights).

        # Generate hash
        h3.reset()
        h3.update(w.tobytes())
        w_hash: int = h3.intdigest()

        duplicate: bool = False
        duplicate_idx: Optional[int] = None
        if w_hash in weight_hash:  # OK to use slow operator (max array length is 128)
            # Check for collisions (hash matches, but other properties don't)
            for i, wh in enumerate(weight_hash):
                if wh == w_hash:
                    weights_out_i = weights_out[i]
                    assert weights_out_i is not None
                    weight_ptrs_i = weight_ptrs[i]
                    assert weight_ptrs_i is not None
                    if w.shape == weights_in[weight_ptrs_i].shape \
                       and (quantization[ll] == quantization[weight_ptrs_i]
                            and processor_map[ll] == processor_map[weight_ptrs_i]
                            or kind == 'bias') \
                       and np.array_equal(weights_out_i, w):
                        duplicate_idx = i
                        duplicate = True
                        break

        i = len(weights_out)
        if not duplicate:
            weight_hash.append(w_hash)
            weight_ptrs.append(i)
            weights_out.append(w)
        else:
            weight_hash.append(None)
            weight_ptrs.append(duplicate_idx)
            weights_out.append(None)
            n += 1
            saved_weight_bytes += reduce(opr.mul, w.shape) * abs(quantization[ll]) // 8

    if n > 0 and state.verbose:
        nprint(f'Deduplication eliminated {"bias values" if kind == "bias" else "weights"} '
               f'for {n} {plural(n, "layer")} ({saved_weight_bytes:,} bytes)')

    return weight_ptrs, weights_out
