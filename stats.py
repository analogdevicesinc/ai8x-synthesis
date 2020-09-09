###################################################################################################
# Copyright (C) Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
###################################################################################################
"""
Statistics for the pure Python computation modules
"""
import tornadocnn as tc


macc = 0  # Hardware multiply-accumulates (Conv2D, etc.)
comp = 0  # Comparisons (ReLU, MaxPool)
add = 0  # Additions (EltwiseAdd, EltwiseSub, AvgPool)
mul = 0  # Multiplications (EltwiseMul)
bitwise = 0  # Bitwise OR/XOR (EltwiseXOR)
# div = 0  # Divisions (BatchNorm, SoftMax)
# exp = 0  # Exponentiations (SoftMax)

sw_macc = 0  # Software multiply-accumulates (FC)
sw_comp = 0  # Software comparisons (ReLU)

true_macc = 0  # Actual MAC ops, ignoring padding
true_sw_macc = 0


def ops():
    """
    Return number of ops computed in the simulator.
    """
    return macc + comp + add + mul + bitwise


def sw_ops():
    """
    Return number of software ops (FC) computed in the simulator.
    """
    return sw_macc + sw_comp


def print_summary(
        factor=1,
        debug=False,
):
    """
    Print ops summary stats.
    """
    print(f'Hardware: {factor * ops():,} ops ({factor * macc:,} macc; {factor * comp:,} '
          f'comp; {factor * add:,} add; '
          f'{factor * mul:,} mul; {factor * bitwise:,} bitwise)')
    if debug:
        print(f'          True MACs: {factor * true_macc:,}')
    if sw_macc:
        print(f'Software: {factor * sw_ops():,} ops ({factor * sw_macc:,} '
              f'macc; {factor * sw_comp:,} comp)')


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
):
    """
    Returns estimated latencies (in cycles) for startup and each layer for a given network setup.
    The return values are an integer (startup cycles) and a list of tuples
    (cycles [integer], detailed description [string]).
    """

    # No support for estimating streaming latency yet
    if any(streaming):
        return None, None

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
