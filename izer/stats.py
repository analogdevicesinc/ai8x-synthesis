###################################################################################################
# Copyright (C) 2019-2021 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
###################################################################################################
"""
Statistics for the pure Python computation modules
"""
import operator
from functools import reduce
from typing import List, Optional, Tuple

from . import state
from . import tornadocnn as tc
from .names import layer_pfx

statsdict = {
    "macc": [0],  # Hardware multiply-accumulates (Conv2D, etc.)
    "comp": [0],  # Comparisons (ReLU, MaxPool)
    "add": [0],  # Additions (EltwiseAdd, EltwiseSub, AvgPool)
    "mul": [0],  # Multiplications (EltwiseMul)
    "bitwise": [0],  # Bitwise OR/XOR (EltwiseXOR)
    # div: [0],  # Divisions (BatchNorm, SoftMax)
    # exp: [0],  # Exponentiations (SoftMax)
    "sw_macc": [0],  # Software multiply-accumulates (FC)
    "sw_comp": [0],  # Software comparisons (ReLU)
    "true_macc": [0],  # Actual MAC ops, ignoring padding
    "true_sw_macc": [0],
}


def get(layer, operation: str) -> int:
    """
    Return the stats of `operation` for a `layer`.
    """
    if operation not in statsdict:
        raise NotImplementedError

    if len(statsdict[operation]) <= layer:
        return 0

    return statsdict[operation][layer]


def ops(layer: Optional[int] = None) -> int:
    """
    Return number of ops computed in the simulator for all layers or a specific layer.
    """
    if layer is None:
        return sum(statsdict["macc"]) + sum(statsdict["comp"]) + sum(statsdict["add"]) \
            + sum(statsdict["mul"]) + sum(statsdict["bitwise"])
    # else:
    return get(layer, "macc") + get(layer, "comp") + get(layer, "add") \
        + get(layer, "mul") + get(layer, "bitwise")


def sw_ops() -> int:
    """
    Return number of software ops (FC) computed in the simulator.
    """
    return sum(statsdict["sw_macc"]) + sum(statsdict["sw_comp"])


def account(
        layer: int,
        operation: str,
        val: int,
) -> None:
    """
    Account for `operation` in `layer`.
    """
    if operation not in statsdict:
        raise NotImplementedError

    dlen = len(statsdict[operation])
    if dlen <= layer:
        statsdict[operation] += [0] * (1 + layer - dlen)

    statsdict[operation][layer] += val


def summary(
        factor: int = 1,
        spaces: int = 0,
        group_bias_max: Optional[int] = None,
) -> str:
    """
    Return ops summary and weight usage statistics.
    """
    # Cache variables locally
    debug = state.debug
    weights = state.weights
    w_size = state.quantization
    bias = state.bias

    sp = ' ' * spaces
    rv = sp + "SUMMARY OF OPS\n"

    rv += f'{sp}Hardware: {factor * ops():,} ops ({factor * sum(statsdict["macc"]):,} macc; ' \
          f'{factor * sum(statsdict["comp"]):,} ' \
          f'comp; {factor * sum(statsdict["add"]):,} add; ' \
          f'{factor * sum(statsdict["mul"]):,} mul; ' \
          f'{factor * sum(statsdict["bitwise"]):,} bitwise)\n'
    if debug:
        rv += f'{sp}          True MACs: {factor * sum(statsdict["true_macc"]):,}\n'
    for ll in range(state.first_layer_used, state.layers):
        rv += f'{sp}  {layer_pfx(ll)}{factor * ops(ll):,} ops ' \
              f'({factor * get(ll, "macc"):,} macc; ' \
              f'{factor * get(ll, "comp"):,} ' \
              f'comp; {factor * get(ll, "add"):,} add; ' \
              f'{factor * get(ll,"mul"):,} mul; ' \
              f'{factor * get(ll, "bitwise"):,} bitwise)\n'

    if sum(statsdict["sw_macc"]) > 0:
        rv += f'{sp}Software: {factor * sw_ops():,} ops ' \
              f'({factor * sum(statsdict["sw_macc"]):,} ' \
              f'macc; {factor * sum(statsdict["sw_comp"]):,} comp)\n'

    assert tc.dev is not None
    if weights is not None and hasattr(tc.dev, 'BIAS_SIZE'):
        kmem = sum(tc.dev.mask_width(proc) * 9 for proc in range(tc.dev.MAX_PROC))
        kmem_used = sum([reduce(operator.mul, e.shape) * abs(w_size[i]) // 8
                         for i, e in enumerate(weights[:len(w_size)]) if e is not None])
        rv += f"\n{sp}RESOURCE USAGE\n" \
              f'{sp}Weight memory: {kmem_used:,} bytes out of {kmem:,} bytes total ' \
              f'({kmem_used * 100.0 / kmem:.0f}%)\n'

        bmem = tc.dev.BIAS_SIZE * tc.dev.P_NUMGROUPS
        if group_bias_max is not None:
            bmem_used = sum(group_bias_max)
        elif bias is not None:
            bmem_used = sum([len(e) for _, e in enumerate(bias) if e is not None])
        else:
            bmem_used = 0
        rv += f'{sp}Bias memory:   {bmem_used:,} bytes out of {bmem:,} bytes total ' \
              f'({bmem_used * 100.0 / bmem:.0f}%)\n'

    return rv


def calc_latency(
        streaming,
        layers,
        eltwise,
        pool,
        pooled_dim,
        in_expand,
        output_chan,
        output_dim,
        input_dim,  # pylint: disable=unused-argument
        padding,
        kernel_size,  # pylint: disable=unused-argument
        debug=False,  # pylint: disable=unused-argument
) -> Tuple[Optional[int], Optional[List]]:
    """
    Returns estimated latencies (in cycles) for startup and each layer for a given network setup.
    The return values are an integer (startup cycles) and a list of tuples
    (cycles [integer], detailed description [string]).
    """

    # No support for estimating streaming latency yet
    if any(streaming):
        return None, None

    assert tc.dev is not None

    lat = []

    for ll in range(layers):
        pad = tc.dev.C_PAD * 2 * (  # Pad cycles * (top + left) * 2 (for bottom + right)
            padding[ll][0] * (2 * padding[ll][1] + pooled_dim[ll][1])
            + padding[ll][1] * pooled_dim[ll][0]
        )
        lk = (eltwise[ll] + 1) * in_expand[ll] * (
            pool[ll][0] * pool[ll][1] * pooled_dim[ll][0] * pooled_dim[ll][1] + pad
        ) + output_dim[ll][0] * output_dim[ll][1] * (output_chan[ll] + tc.dev.C_PAD)
        s = f'Input: eltwise {eltwise[ll] + 1} * in_expand {in_expand[ll]} ' \
            f'* (pool {pool[ll][0]}x{pool[ll][1]}={pool[ll][0] * pool[ll][1]} * ' \
            f'pooled_dim {pooled_dim[ll][0]}x{pooled_dim[ll][1]}' \
            f'={pooled_dim[ll][0] * pooled_dim[ll][1]} + Pad: {tc.dev.C_PAD}*2*' \
            f'({padding[ll][0]}x({padding[ll][1]}+{padding[ll][1]}+{pooled_dim[ll][1]})=' \
            f'{padding[ll][0] * (2 * padding[ll][1] + pooled_dim[ll][1])} + ' \
            f'{padding[ll][1]}x{pooled_dim[ll][0]}={padding[ll][1] * pooled_dim[ll][0]}))=' \
            f'{pad} + ' \
            f'TRAM shift: {tc.dev.C_PAD}x{output_dim[ll][0]}x{output_dim[ll][1]}=' \
            f'{tc.dev.C_PAD * output_dim[ll][0] * output_dim[ll][1]} + ' \
            f'Output: {output_dim[ll][0]}x{output_dim[ll][1]}x{output_chan[ll]}=' \
            f'{output_dim[ll][0] * output_dim[ll][1] * output_chan[ll]}'

        lat.append((lk, s))

    return tc.dev.C_START, lat
