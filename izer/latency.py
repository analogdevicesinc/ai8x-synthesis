###################################################################################################
# Copyright (C) 2022 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
###################################################################################################
"""
Latency calculations
"""
from typing import Tuple

from izer import tornadocnn as tc


def calculate(
        input_chan: int,
        input_dim: Tuple[int, int],
        pool: Tuple[int, int],
        pool_stride: Tuple[int, int],
        pooled_dim: Tuple[int, int],
        multipass: int,
        output_chan: int,
        output_dim: Tuple[int, int],
        kernel_size: Tuple[int, int],
        padding: Tuple[int, int],
        num_elements: int,
        pool_first: bool,
        passthrough: bool,
        pass_out_chan: int,
        flatten: bool,
        streaming: bool,
        kern_offs: int,
) -> Tuple[int, str]:
    """
    Calculate the latency in cycles for a single layer with the arguments given.
    """
    # Input
    img_rows, img_cols = input_dim  # type: Tuple[int, int]
    pool_rows, pool_cols = pool  # type: Tuple[int, int]
    pool_row_stride, pool_col_stride = pool_stride  # type: Tuple[int, int]
    img_pooled_rows, img_pooled_cols = pooled_dim  # type: Tuple[int, int]
    row_pad, col_pad = padding  # type: Tuple[int, int]
    kern_rows, kern_cols = kernel_size  # type: Tuple[int, int]
    output_rows, output_cols = output_dim  # type: Tuple[int, int]

    s: str = '                 Channels C     Rows H  Columns W\n' \
        f'Input dimensions{input_chan:11}{img_rows:11}{img_cols:11}\n' \
        f'Pooling{pool_rows:31}{pool_cols:11}\n' \
        f'Pooling stride{pool_row_stride:24}{pool_col_stride:11}\n' \
        f'Pooled dimensions{img_pooled_rows:21}{img_pooled_cols:11}\n' \
        f'Padding{row_pad:31}{col_pad:11}\n' \
        f'Kernel size{kern_rows:27}{kern_cols:11}\n' \
        f'Output dimensions{output_chan:10}{output_rows:11}{output_cols:11}\n\n'

    rd_state: int = 1
    tram_wrt: int = 1
    wrt_state: int = 1
    dummy_clock: int = 1
    flatten_adj: int = 1 if flatten else 0  # FIXME
    meq_one: int = 1 if output_chan == 1 and multipass == 1 else 0

    s += f'Multipass{multipass:18}\n' \
        f'NumElements{num_elements:16}\n' \
        f'PoolFirst{pool_first:18}\n' \
        f'Passthrough{passthrough:16}\n' \
        f'RdState{rd_state:20}\n' \
        f'TRAMWrt{tram_wrt:20}\n' \
        f'WrtState{wrt_state:19}\n' \
        f'DummyClock{dummy_clock:17}\n' \
        f'PassOutChan{pass_out_chan:16}\n' \
        f'MeqOne{meq_one:21}\n' \
        f'FlattenAdj{flatten_adj:17}\n\n'

    pre_read: int = 0
    # Input processing
    if pool_first:
        in_pad_mp: int = multipass * num_elements * (rd_state + tram_wrt)
        in_dat_mp: int = multipass * num_elements * ((pool[0] * pool[1] * rd_state) + tram_wrt)
        mp_multiplier: int = 1

        s += f'InPadMPPoolElt{in_pad_mp:13}\n' \
            f'InDatMPPoolElt{in_dat_mp:13}\n\n'
    else:  # Implies pooling > 1,1
        assert tc.dev is not None
        if tc.dev.REQUIRE_MP_KERNOFF_MULTIWRITE \
           and (multipass > 1 or num_elements > 1):
            pre_read = ((pooled_dim[0] + 2*row_pad - kern_cols + 1) * col_pad
                        + (pooled_dim[1] + 2*col_pad - kern_rows + 1) * row_pad
                        - row_pad * col_pad) * kern_offs
            s += f'PreRead{pre_read:20}\n'
        in_pad_mp = multipass * (rd_state + tram_wrt)
        in_dat_mp = multipass * ((num_elements * rd_state * pool[0] * pool[1]) + tram_wrt)
        mp_multiplier = multipass

        s += f'InPadMPEltPool{in_pad_mp:13}\n' \
            f'InDatMPEltPool{in_dat_mp:13}\n\n'

    col: int = 2 * col_pad * (in_pad_mp + wrt_state)
    row: int = mp_multiplier * img_pooled_cols * (in_pad_mp + wrt_state)

    inp_pad_row: int = col + row

    s += '                      Total    Col Pad   Row Data\n' \
        f'First Row Pad{inp_pad_row:14}{col:11}{row:11}\n'

    row = mp_multiplier * img_pooled_cols * (in_dat_mp + wrt_state)
    inp_dat_row_no_conv: int = col + row

    s += f'Row Data (No Conv){inp_dat_row_no_conv:9}{col:11}{row:11}\n'

    left_col_pad: int = col_pad * (in_pad_mp + wrt_state)
    data_no_conv: int = mp_multiplier * (kern_cols - 1 - col_pad) * (in_dat_mp + wrt_state)
    data_conv: int = mp_multiplier * (output_cols - col_pad) * in_dat_mp
    right_pad: int = in_pad_mp * col_pad

    inp_dat_row_conv: int = left_col_pad + data_no_conv + data_conv + right_pad

    s += '                      Total   Left Pad   Data NoC  Data Conv  Right Pad\n' \
        f'Row Data{inp_dat_row_conv:19}{left_col_pad:11}' \
        f'{data_no_conv:11}{data_conv:11}{right_pad:11}\n'

    data_no_conv = mp_multiplier * (kern_cols - 1 - col_pad) * (in_pad_mp + wrt_state)
    data_conv = mp_multiplier * (output_cols - col_pad) * in_pad_mp

    inp_pad_bottom_row: int = left_col_pad + data_no_conv + data_conv + right_pad

    s += f'Bottom Pad Row{inp_pad_bottom_row:13}{left_col_pad:11}' \
        f'{data_no_conv:11}{data_conv:11}{right_pad:11}\n\n'

    # Output processing
    if passthrough:
        out_dat: int = multipass * img_pooled_rows * img_pooled_cols
    else:
        out_dat = multipass * output_cols * output_chan + output_chan * meq_one

    s += f'Data Row{out_dat:19}\n\n'

    # Frame equation
    if passthrough:
        total: int = img_pooled_rows * img_pooled_cols * (in_dat_mp + multipass * pass_out_chan)
    else:
        total = flatten_adj + pre_read + row_pad * inp_pad_row + \
            ((kern_rows - 1 - row_pad) * inp_dat_row_no_conv) + \
            (inp_dat_row_conv + out_dat) * (img_pooled_rows - (kern_rows - 1 - row_pad)) + \
            row_pad * (inp_pad_bottom_row + out_dat)

    if not streaming:
        s += f'Layer Subtotal{total:13}\n'
    else:
        total = (15 * total + 9) // 10  # 1.5x regular layer
        s += f'Layer Subtotal{total:13} (streaming estimate)\n'

    return total, s
