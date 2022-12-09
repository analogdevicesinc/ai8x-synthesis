#!/usr/bin/env python3
###################################################################################################
# Copyright (C) 2019-2022 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
###################################################################################################
"""
Statistics for the pure Python computation modules
"""
import os
import sys
from typing import List, Tuple

# Allow test to run outside of pytest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import izer.tornadocnn as tc  # noqa: E402 pylint: disable=wrong-import-position
from izer import latency  # noqa: E402 pylint: disable=wrong-import-position
from izer import op  # noqa: E402 pylint: disable=wrong-import-position


def calc_dim(
    input_dim: Tuple[int, int],
    pool: Tuple[int, int],
    pool_stride: Tuple[int, int],
    kernel_size: Tuple[int, int],
    padding: Tuple[int, int],
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """
    Calculated pooled dimensions and output dimensions
    """
    pd: Tuple[int, int] = ((input_dim[0] + pool_stride[0] - pool[0]) // pool_stride[0],
                           (input_dim[1] + pool_stride[1] - pool[1]) // pool_stride[1])
    # print('Calculated pooled dimensions:', pd)

    od: Tuple[int, int] = (pd[0] - (kernel_size[0] - 1) - 1 + 2 * padding[0] + 1,
                           pd[1] - (kernel_size[1] - 1) - 1 + 2 * padding[1] + 1)
    # print('Calculated output dimensions:', od, '\n')

    return pd, od


def wrapper(
        input_chan: List[int],
        input_dim: List[Tuple[int, int]],
        pool: List[Tuple[int, int]],
        pool_stride: List[Tuple[int, int]],
        in_expand: List[int],
        output_chan: List[int],
        kernel_size: List[Tuple[int, int]],
        padding: List[Tuple[int, int]],
        operands: List[int],
        pool_first: List[bool],
        operator: List[int],
        output_width: List[int],  # pylint: disable=unused-argument,
        timeslots: List[int],
        flatten: List[bool],
        streaming: List[bool],
        name: str = '',
) -> int:
    """
    Wrapper for latency.calculate() that calculates pooled dimensions and
    output dimensions.
    """
    print('------------', name)

    pooled_dim = []
    output_dim = []
    kern_count = []
    for ll, _ in enumerate(input_chan):
        idim: Tuple[int, int] = (input_dim[ll][0] * 2, input_dim[ll][1] * 2) \
            if operator[ll] == op.CONVTRANSPOSE2D else input_dim[ll]
        pd, od = calc_dim(
            input_dim=idim,
            pool=pool[ll],
            pool_stride=pool_stride[ll],
            padding=padding[ll],
            kernel_size=kernel_size[ll],
        )
        pooled_dim.append(pd)
        output_dim.append(od)
        kern_count.append(in_expand[ll] * output_chan[ll])

    start = total = 1
    print(f'START{start:22}\n')

    for ll, in_ch in enumerate(input_chan):
        lat, comment = latency.calculate(
            input_chan=in_ch,
            input_dim=input_dim[ll],
            pool=pool[ll],
            pool_stride=pool_stride[ll],
            pooled_dim=pooled_dim[ll],
            multipass=in_expand[ll],
            output_chan=kern_count[ll] // in_expand[ll]
            if operator[ll] != op.NONE else output_chan[ll],
            output_dim=output_dim[ll],
            kernel_size=kernel_size[ll],
            padding=padding[ll],
            num_elements=operands[ll],
            pool_first=pool_first[ll],
            passthrough=operator[ll] == op.NONE,
            pass_out_chan=timeslots[ll],
            flatten=flatten[ll],
            streaming=streaming[ll],
            kern_offs=0,
        )
        total += lat
        print(f'Layer {ll} Detail:\n')
        print(comment)

    print('===========================\n'
          f'TOTAL ALL LAYERS{total:11}\n')

    return total


def test_cycles() -> None:
    """
    Main program to test layer cycle calculations
    """
    tc.dev = tc.get_device(85)

    c = wrapper(
        input_chan=[64],
        input_dim=[(20, 20)],
        pool=[(2, 2)],
        pool_stride=[(2, 2)],
        in_expand=[1],
        output_chan=[16],
        kernel_size=[(3, 3)],
        padding=[(2, 2)],
        operands=[1],
        pool_first=[True],
        operator=[op.CONV2D],
        output_width=[8],
        timeslots=[1],
        flatten=[False],
        streaming=[False],
        name='ai85-kmax_bmax_dmax-64x20x20l_max2x2s2p2m16',
    )
    assert c == 3049

    c = wrapper(
        input_chan=[3],
        input_dim=[(65, 65)],
        pool=[(16, 16)],
        pool_stride=[(4, 4)],
        in_expand=[1],
        output_chan=[32],
        kernel_size=[(3, 3)],
        padding=[(1, 1)],
        operands=[1],
        pool_first=[True],
        operator=[op.CONV2D],
        output_width=[8],
        timeslots=[1],
        flatten=[False],
        streaming=[False],
        name='ai85-pool-16x16s4-3x65x65l_max16x16s',
    )
    assert c == 49010

    c = wrapper(
        input_chan=[3],
        input_dim=[(32, 32)],
        pool=[(1, 1)],
        pool_stride=[(1, 1)],
        in_expand=[1],
        output_chan=[64],
        kernel_size=[(3, 3)],
        padding=[(1, 1)],
        operands=[1],
        pool_first=[True],
        operator=[op.CONV2D],
        output_width=[8],
        timeslots=[1],
        flatten=[False],
        streaming=[False],
        name='ai85-cifar-3x32x32b_0x0s1p1m60_relu',
    )
    assert c == 67981

    c = wrapper(
        input_chan=[3],
        input_dim=[(32, 32)],
        pool=[(1, 1)],
        pool_stride=[(1, 1)],
        in_expand=[1],
        output_chan=[64],
        kernel_size=[(3, 3)],
        padding=[(0, 0)],
        operands=[1],
        pool_first=[True],
        operator=[op.CONV2D],
        output_width=[8],
        timeslots=[1],
        flatten=[False],
        streaming=[False],
        name='ai85-cifar-3x32x32b_2x2s2p0m64_relu',
    )
    assert c == 59773

    c = wrapper(
        input_chan=[3],
        input_dim=[(32, 32)],
        pool=[(2, 2)],
        pool_stride=[(2, 2)],
        in_expand=[1],
        output_chan=[64],
        kernel_size=[(3, 3)],
        padding=[(0, 0)],
        operands=[1],
        pool_first=[True],
        operator=[op.CONV2D],
        output_width=[8],
        timeslots=[1],
        flatten=[False],
        streaming=[False],
        name='ai85-cifar-3x32x32b_2x2s2p0m64_relu',
    )
    assert c == 13885

    c = wrapper(
        input_chan=[128],
        input_dim=[(8, 8)],
        pool=[(2, 2)],
        pool_stride=[(2, 2)],
        in_expand=[2],
        output_chan=[16],
        kernel_size=[(3, 3)],
        padding=[(1, 1)],
        operands=[1],
        pool_first=[True],
        operator=[op.CONV2D],
        output_width=[8],
        timeslots=[1],
        flatten=[False],
        streaming=[False],
        name='ai85-widein-128x8x8l_2x2s2p1m16',
    )
    assert c == 773

    c = wrapper(
        input_chan=[128],
        input_dim=[(8, 8)],
        pool=[(1, 1)],
        pool_stride=[(1, 1)],
        in_expand=[2],
        output_chan=[16],
        kernel_size=[(3, 3)],
        padding=[(1, 1)],
        operands=[1],
        pool_first=[True],
        operator=[op.CONV2D],
        output_width=[8],
        timeslots=[1],
        flatten=[False],
        streaming=[False],
        name='ai85-widein-128x8x8l_0x0s1p1m16',
    )
    assert c == 2485

    c = wrapper(
        input_chan=[128],
        input_dim=[(8, 8)],
        pool=[(1, 1)],
        pool_stride=[(1, 1)],
        in_expand=[2],
        output_chan=[16],
        kernel_size=[(3, 3)],
        padding=[(0, 0)],
        operands=[1],
        pool_first=[True],
        operator=[op.CONV2D],
        output_width=[8],
        timeslots=[1],
        flatten=[False],
        streaming=[False],
        name='ai85-widein-128x8x8l_0x0s1p0m16',
    )
    assert c == 1437

    c = wrapper(
        input_chan=[1],
        input_dim=[(28, 28)],
        pool=[(1, 1)],
        pool_stride=[(1, 1)],
        in_expand=[1],
        output_chan=[8],
        kernel_size=[(3, 3)],
        padding=[(1, 1)],
        operands=[1],
        pool_first=[True],
        operator=[op.CONV2D],
        output_width=[8],
        timeslots=[1],
        flatten=[False],
        streaming=[False],
        name='ai85-mnist-extrasmall-1x28x28b_0x0s1p1m8_relu',
    )
    assert c == 8189

    c = wrapper(
        input_chan=[4],
        input_dim=[(12, 12)],
        pool=[(1, 1)],
        pool_stride=[(1, 1)],
        in_expand=[1],
        output_chan=[4],
        kernel_size=[(1, 1)],
        padding=[(0, 0)],
        operands=[2],
        pool_first=[True],
        operator=[op.NONE],
        output_width=[8],
        timeslots=[1],
        flatten=[False],
        streaming=[False],
        name='ai85-eltwise-add-4x12x12l_0x0s1p0m4',
    )
    assert c == 721

    c = wrapper(
        input_chan=[3],
        input_dim=[(7, 9)],
        pool=[(1, 1)],
        pool_stride=[(1, 1)],
        in_expand=[1],
        output_chan=[64],
        kernel_size=[(3, 3)],
        padding=[(1, 1)],
        operands=[1],
        pool_first=[True],
        operator=[op.CONVTRANSPOSE2D],
        output_width=[8],
        timeslots=[1],
        flatten=[False],
        streaming=[False],
        name='ai85-upscale-3x7x9l_0x0s1p1m64',
    )
    assert c == 16837

    c = wrapper(
        input_chan=[12],
        input_dim=[(4, 4)],
        pool=[(1, 1)],
        pool_stride=[(1, 1)],
        in_expand=[1],
        output_chan=[10],
        kernel_size=[(1, 1)],
        padding=[(0, 0)],
        operands=[1],
        pool_first=[True],
        operator=[op.CONV2D],
        output_width=[8],
        timeslots=[1*4*4],
        flatten=[False],
        streaming=[False],
        name='ai85-mlpflatten192to10-12x4x4lf_0x0s1p0m10',
    )
    assert c == 193

    c = wrapper(
        input_chan=[4],
        input_dim=[(12, 12)],
        pool=[(1, 1)],
        pool_stride=[(1, 1)],
        in_expand=[2],
        output_chan=[4],
        kernel_size=[(3, 3)],
        padding=[(0, 0)],
        operands=[2],
        pool_first=[True],
        operator=[op.NONE],
        output_width=[8],
        timeslots=[1],
        flatten=[False],
        streaming=[False],
        name='ai85-eltwise-add-multipass-4x12x12l_0x0s1p0m4',
    )
    assert c == 1441

    c = wrapper(
        input_chan=[79],
        input_dim=[(11, 11)],
        pool=[(1, 1)],
        pool_stride=[(1, 1)],
        in_expand=[2],
        output_chan=[1],
        kernel_size=[(3, 3)],
        padding=[(0, 0)],
        operands=[1],
        pool_first=[True],
        operator=[op.CONV2D],
        output_width=[8],
        timeslots=[1],
        flatten=[False],
        streaming=[False],
        name='ai85-widein-q4-79x11x11l_0x0s1p0m1',
    )
    assert c == 687

    c = wrapper(
        input_chan=[8],
        input_dim=[(17, 17)],
        pool=[(2, 2)],
        pool_stride=[(2, 2)],
        in_expand=[1],
        output_chan=[8],
        kernel_size=[(1, 1)],
        padding=[(0, 0)],
        operands=[2],
        pool_first=[False],
        operator=[op.NONE],
        output_width=[8],
        timeslots=[2],
        flatten=[False],
        streaming=[False],
        name='ai85-eltwise-poolafter-8x17x17l_max2x2s2p0m8',
    )
    assert c == 705

    c = wrapper(
        input_chan=[12],
        input_dim=[(8, 8)],
        pool=[(1, 1)],
        pool_stride=[(1, 1)],
        in_expand=[1],
        output_chan=[10],
        kernel_size=[(1, 1)],
        padding=[(0, 0)],
        operands=[1],
        pool_first=[True],
        operator=[op.CONV2D],
        output_width=[8],
        timeslots=[1],
        flatten=[True],
        streaming=[False],
        name='ai85-mlpflatten768to10-12x8x8lf_0x0s1p0m10',
    )
    assert c == 770

    c = wrapper(
        input_chan=[1],
        input_dim=[(1, 1)],
        pool=[(1, 1)],
        pool_stride=[(1, 1)],
        in_expand=[1],
        output_chan=[1],
        kernel_size=[(3, 3)],
        padding=[(1, 1)],
        operands=[1],
        pool_first=[True],
        operator=[op.CONV2D],
        output_width=[8],
        timeslots=[1],
        flatten=[False],
        streaming=[False],
        name='ai85-singlebyte-hwc-1x1x1l_0x0s1p1m1',
    )
    assert c == 29

    c = wrapper(
        input_chan=[12],
        input_dim=[(1, 1)],
        pool=[(1, 1)],
        pool_stride=[(1, 1)],
        in_expand=[1],
        output_chan=[17],
        kernel_size=[(1, 1)],
        padding=[(0, 0)],
        operands=[1],
        pool_first=[True],
        operator=[op.CONV2D],
        output_width=[8],
        timeslots=[1],
        flatten=[False],
        streaming=[False],
        name='ai85-mlpflatten12to17-12x1x1lf_0x0s1p0m17',
    )
    assert c == 20

    c = wrapper(
        input_chan=[5],
        input_dim=[(31, 7)],
        pool=[(1, 1)],
        pool_stride=[(1, 1)],
        in_expand=[1],
        output_chan=[11],
        kernel_size=[(3, 3)],
        padding=[(1, 1)],
        operands=[1],
        pool_first=[True],
        operator=[op.CONV2D],
        output_width=[8],
        timeslots=[1],
        flatten=[False],
        streaming=[False],
        name='test-conv1d-dilation7-pad0-5x211l_0s1p0m11',
    )
    assert c == 3062

    print('*** PASS ***')


if __name__ == '__main__':
    test_cycles()
