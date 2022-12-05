###################################################################################################
# Copyright (C) 2022 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
###################################################################################################
"""
Define memories.
"""
import numpy as np

from . import state
from . import tornadocnn as tc
from .eprint import eprint
from .names import layer_pfx, layer_str

_UNUSED = -(2**63)


def allocate():
    """
    Allocate an empty memory map for all data memory.
    """
    return np.full((tc.dev.P_NUMGROUPS * tc.dev.MEM_SIZE * tc.dev.P_NUMPRO // tc.dev.P_SHARED),
                   dtype=np.int64, fill_value=_UNUSED)


def idx(offs):
    """
    Return the data memory array index given a byte offset `offs`.
    """
    group, memoffs = divmod(offs - tc.dev.C_SRAM_BASE, tc.dev.C_GROUP_OFFS)
    return group * tc.dev.MEM_SIZE * tc.dev.P_NUMPRO // tc.dev.P_SHARED + (memoffs >> 2)


def validate(arr, offs, val=None):
    """
    Check whether we're overwriting location `offs` in array `arr`.
    """
    old_ll, old_c, old_row, old_col = unpack(arr, offs)
    if old_ll is not None:
        if val is None:
            nstr = ''
        else:
            (ll, c, row, col) = val
            nstr = layer_pfx(ll) + f'CHW={c},{row},{col} - '
        eprint(f'{nstr}Overwriting location 0x{offs:08x}, previously used by layer '
               f'{layer_str(old_ll)}, CHW={old_c},{old_row},{old_col}.',
               error=not state.no_error_stop)


def store(arr, offs, val, check_overwrite=False):
    """
    Pack layer/channel/row/column into int64 value

    On MAX78002: Used here:
    ll:   8 bits    16 bits   (signed, can be negative for input layer)
    c:   10 bits    16 bits   (unsigned)
    row: 11 bits    16 bits   (unsigned)
    col: 11 bits    16 bits   (unsigned)
         -------    -------
         40 bits    64 bits
    """
    if check_overwrite:
        validate(arr, offs, val)
    (ll, c, row, col) = val
    try:
        arr[idx(offs)] = (ll << 48) | (c << 32) | (row << 16) | col
    except IndexError:
        eprint(f'Data memory overflow in layer {layer_str(ll)} for '
               f'offset 0x{offs:08x}, c={c}, row={row}, col={col}.')


def used(arr, offs):
    """
    Check whether memory at offset `offs` is unused.
    """
    return arr[idx(offs)] != _UNUSED


def unpack(arr, offs):
    """
    Unpack int64 value from array `arr` offset `offs` into layer/channel/row/column.
    The function returns (None, None, None, None) if the offset does not contain information.
    """
    ll = arr[idx(offs)]
    if ll == _UNUSED:
        return None, None, None, None
    col = ll & 0xffff
    ll >>= 16
    row = ll & 0xffff
    ll >>= 16
    c = ll & 0xffff
    ll >>= 16
    return ll, c, row, col


def combine(a, b):
    """
    Combine two memory maps `a` and `b`; use the first if it's used, else the second.
    """
    mask = a == _UNUSED
    a[mask] = b[mask]
