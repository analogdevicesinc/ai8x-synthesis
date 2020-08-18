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
        input_dim,  # pylint: disable=unused-argument
        padding,
        kernel_size,  # pylint: disable=unused-argument
        debug=False,  # pylint: disable=unused-argument
):
    """
    Returns estimated latencies (in cycles) for startup, each layer, and total for a
    given network setup.
    The return value is a list of tuples (cycles [integer], detailed description [string])
    """

    # No support for estimating streaming latency yet
    if any(streaming):
        return None

    lat = [(tc.dev.C_START, '')]
    total = tc.dev.C_START

    for k in range(layers):
        s = f'conv: (eltwise {eltwise[k] + 1} * pool {pool[k][0]}x{pool[k][1]}=' \
            f'{pool[k][0] * pool[k][1]} * ' \
            f'in_expand {in_expand[k]} + in_expand {in_expand[k]} + ' \
            f'output_chan {output_chan[k]}) * ' \
            f'pooled_dim {pooled_dim[k][0]}x{pooled_dim[k][1]}' \
            f'={pooled_dim[k][0] * pooled_dim[k][1]}'
        lk = ((eltwise[k] + 1) * pool[k][0] * pool[k][1] * in_expand[k]
              + in_expand[k] + output_chan[k]) * pooled_dim[k][0] * pooled_dim[k][1]
        if padding[k][0] > 0 or padding[k][1] > 0:
            s += f' + pad: padding {padding[k][0]}x{padding[k][1]}, ' \
                 f'pooled_dim {pooled_dim[k][0]}x{pooled_dim[k][1]}=' \
                 f'{pooled_dim[k][0]*pooled_dim[k][1]}'
            lk += tc.dev.C_POOL * (2 * padding[k][0] * (pooled_dim[k][1] + 2 * padding[k][1])
                                   + 2 * padding[k][1] * pooled_dim[k][0])
        lat.append((lk, s))
        total += lk

    lat.append((total, ''))
    return lat
