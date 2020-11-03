###################################################################################################
# Copyright (C) Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
###################################################################################################
"""
Kernel related functions
"""
import math
import sys

import numpy as np

import op
import rv
import tornadocnn as tc
from eprint import eprint, eprint_noprefix
from utils import ffs, fls, popcount


_INVALID_VALUE = -(2**63)
_WORDS_PER_KERNEL = 3


def print_map(
        layers,
        kmap,
        print_fn=print,
):
    """
    Print map of all used kernels in kernel map `kmap`. `layers` describes the number of layers
    in the network and is used to align the map.
    """
    width = int(math.log10(layers)) + 1
    if width > 1:
        width += 1  # Add space if wider than a single character

    assert kmap.shape[0] == tc.dev.MAX_PROC
    assert kmap.shape[1] == tc.dev.MASK_WIDTH_LARGE

    print_fn('-' * tc.dev.MASK_WIDTH_LARGE * width)
    for row in range(tc.dev.MAX_PROC):
        for col in range(tc.dev.mask_width(row)):
            val = kmap[row][col]
            if val == _INVALID_VALUE:
                val = 'X'
            print_fn('{:>{w}}'.format(val, w=width), end='')
        print_fn('')
    print_fn('-' * tc.dev.MASK_WIDTH_LARGE * width)


def load(  # pylint: disable=too-many-branches,too-many-statements
        verbose,
        embedded_code,
        device,
        apb,
        start_layer,
        layers,
        operator,
        kernel,
        kernel_size,
        quantization,
        processor_map,
        output_processor_map,
        input_chan,
        output_chan,
        out_expand,
        out_expand_thresh,
        in_expand,
        in_expand_thresh,
        flatten=False,
        mexpress=False,
        verify=False,
        riscv_flash=False,
        quad=False,
        debug=False,
        blocklevel=False,
        legacy_kernels=False,
        calcx4=False,
):
    """
    Stack `kernel` values and write them to C code (for `embedded_code` if `True` or
    RTL simulation). The output is written to the `apb` object.
    Input is configured with `kernel_size`, `quantization`, `layers`, `processor_map`,
    `output_processor_map`, `input_chan`, `output_chan`, `out_expand` and `out_expand_thresh`.
    When `mexpress` is `True`, the function uses the memcpy()-friendly hardware functionality to
    reduce the number of transfers. When `verify` is also true (mexpress mode only), kernels are
    read back and compared.
    This function returns the kernel offsets and the kernel lengths for all layers.
    """
    # Kernels: Stack kernels; write only the kernels needed
    proc_kern_max = [0] * tc.dev.MAX_PROC
    kern_offs = [0] * layers
    kern_len = [0] * layers
    kernel_map = np.full((tc.dev.MAX_PROC, tc.dev.MASK_WIDTH_LARGE),
                         _INVALID_VALUE, dtype=np.int64)
    kernels_used = np.zeros((tc.dev.MAX_PROC, tc.dev.MASK_WIDTH_LARGE), dtype=np.int64)
    kernel_data = np.zeros((tc.dev.MAX_PROC, tc.dev.MASK_WIDTH_LARGE, 9), dtype=np.int8)
    # There are four 32-bit words per 9-byte kernel.
    # The value map is initialized with zeros so we can later ignore unused entries and use
    # memcpy() on initialized and uninitialized data.
    kernel_values = np.zeros((tc.dev.MAX_PROC, tc.dev.MASK_WIDTH_LARGE * _WORDS_PER_KERNEL),
                             dtype=np.int64)
    if debug:
        print('\nLoading Kernels...')

    if calcx4 and not tc.dev.SUPPORT_CALCX4:
        eprint('--calcx4 is not supported on this device.')
        sys.exit(1)
    assert not ((embedded_code or mexpress) and calcx4)  # FIXME Add support later

    for ll in range(start_layer, layers):
        if operator[ll] not in [op.CONV1D, op.CONV2D, op.CONVTRANSPOSE2D]:
            kern_len[ll] = 0
            kern_offs[ll] = 0
            continue

        if flatten[ll]:
            kernel_reshaped = kernel[ll].reshape(
                output_chan[ll] * input_chan[ll],
                -1,
                kernel_size[ll][0],
                kernel_size[ll][1],
            )
        else:
            kernel_reshaped = kernel[ll]

        first_proc = ffs(processor_map[ll])
        last_proc = fls(processor_map[ll])
        ch = 0
        m = 0
        for p in range(first_proc, last_proc+1):
            if (processor_map[ll] >> p) & 1 == 0:
                # Unused processor
                continue
            # Get highest offset for all used processors
            kern_offs[ll] = max(proc_kern_max[p], kern_offs[ll])

        ksize = kernel_size[ll][0] * kernel_size[ll][1]
        qfactor = 8 // quantization[ll]
        # Determine the number of kernels that need to be programmed. Since each instance
        # spans 4 processors, kernels for all instances that have a single processor enabled
        # need to be written, i.e. round down the first. The last does not need to be rounded
        # up because hardware takes care of it.
        next_layer_map = output_processor_map[ll]
        # When using kernels smaller than 8 bit, round up to the next 8-bit boundary
        # Gaps are accounted for like any other kernel.
        kern_len[ll] = 1 + quantization[ll] * \
            (fls(next_layer_map) - ffs(next_layer_map)) // 8
        # This extends the kernels to the right on AI85 for input and output expansion
        if output_chan[ll] > tc.dev.MAX_PROC:
            kern_len[ll] = (kern_len[ll] + tc.dev.P_SHARED-1) & ~(tc.dev.P_SHARED-1)
        kern_len[ll] *= out_expand[ll] * in_expand[ll]
        if not legacy_kernels and flatten[ll]:
            kern_len[ll] *= kernel_reshaped.shape[1]
            kern_len[ll] -= (out_expand[ll] * popcount(next_layer_map) - output_chan[ll]) \
                * kernel_reshaped.shape[1] * 8 // (ksize * quantization[ll])
        if device != 84:
            # Pack kernels when using 1D convolutions, or 1x1 kernels
            kern_len[ll] = (kern_len[ll] * ksize + 8) // 9
        if ll == 0 and quad:
            kern_len[0] = (kern_len[0] + 3) // 4

        # We don't have to use dummy columns if there's space available on the left
        kern_offs[ll] = \
            max(0, kern_offs[ll] - (((ffs(next_layer_map) % tc.dev.P_SHARED)
                                     + qfactor - 1) // qfactor))
        # The kernel offset needs to start at a multiple of 4.
        kern_offs[ll] = (kern_offs[ll] + tc.dev.P_SHARED-1) & ~(tc.dev.P_SHARED-1)
        if kern_offs[ll] + kern_len[ll] > tc.dev.mask_width(p):
            eprint(f'\nKernel memory exceeded at layer {ll}; offset: {kern_offs[ll]}, '
                   f'needed: {kern_len[ll]}.'
                   '\n\nKernel map so far:')
            print_map(layers, kernel_map, print_fn=eprint_noprefix)
            sys.exit(1)

        proc_mask = 2**qfactor - 1
        # Start at the first used instance
        this_map_init = next_layer_map >> ffs(next_layer_map)
        start_col = ffs(next_layer_map) % tc.dev.P_SHARED  # First target column

        for p in range(first_proc, last_proc+1):
            if (processor_map[ll] >> p) & 1 == 0:
                # Unused source processor
                continue
            col_target = start_col
            for expand in range(out_expand[ll]):
                this_map = this_map_init
                if ll == 0 and quad:
                    col = expand * (out_expand_thresh[ll] + 3) // 4
                    stop_col = col + (out_expand_thresh[ll] + 3) // 4
                else:
                    col = expand * out_expand_thresh[ll]
                    stop_col = col + out_expand_thresh[ll]
                while col < stop_col:
                    # Skip over unused bits in the target processor map
                    # (unused means 1 bit for 8-bit weights, 2 for 4-bit weights, etc.)
                    if this_map != 0:
                        while this_map & proc_mask == 0:
                            assert this_map != 0
                            col_target += 1  # Completely skip
                            this_map >>= qfactor  # and slide forward
                    this_mask = this_map & proc_mask
                    this_map >>= qfactor

                    if ll == 0 and quad:
                        src_offs = ch + (m - p // 16) * input_chan[ll]
                    else:
                        src_offs = ch + m * input_chan[ll]
                    if ll > 0 or not quad or (m % 4 == p // 16):
                        for ie in range(in_expand[ll]):
                            mask = this_mask

                            def add_kernel_data(ll, p, col_target, b):
                                col = kern_offs[ll] + col_target
                                if col >= tc.dev.mask_width(p):
                                    eprint(f'\nKernel memory exceeded in layer {ll}.'
                                           '\n\nKernel map so far:')
                                    print_map(layers, kernel_map, print_fn=eprint_noprefix)
                                    sys.exit(1)

                                if kernels_used[p][col] == 0:  # Update kernel map
                                    assert kernel_map[p][col] == _INVALID_VALUE
                                    kernel_map[p][col] = ll

                                assert kernels_used[p][col] <= 8
                                kernel_data[p][col][8 - kernels_used[p][col]] = b & 0xff
                                kernels_used[p][col] += 1

                                if kernels_used[p][col] == 9:  # Flush
                                    col_target += 1  # Write 1

                                return col_target

                            n = 0
                            if src_offs < len(kernel_reshaped):
                                if not flatten[ll]:
                                    k = np.zeros_like(kernel_reshaped[src_offs].flatten())
                                    for i in range(qfactor):
                                        if m < output_chan[ll]:
                                            # Cycle through phases
                                            idx = n + ie * qfactor
                                            koffs = src_offs + (idx % in_expand[ll]) \
                                                * in_expand_thresh[ll] \
                                                + (idx // in_expand[ll]) \
                                                * input_chan[ll]
                                            if koffs < len(kernel_reshaped):
                                                this_kern = kernel_reshaped[koffs].flatten() \
                                                    & (2**quantization[ll]-1)
                                                k |= this_kern << (i * quantization[ll])
                                            n += 1
                                        mask >>= 1
                                else:
                                    kl = (len(kernel_reshaped[src_offs]) + qfactor - 1) // qfactor
                                    k = np.zeros(kl, dtype=np.int64)
                                    if m < output_chan[ll]:
                                        # Cycle through phases
                                        idx = n + ie * qfactor
                                        koffs = src_offs + (idx % in_expand[ll]) \
                                            * in_expand_thresh[ll] \
                                            + (idx // in_expand[ll]) \
                                            * input_chan[ll]
                                        if koffs < len(kernel_reshaped):
                                            this_kern = kernel_reshaped[koffs].flatten()
                                            if len(this_kern) % qfactor != 0:
                                                this_kern = np.append(
                                                    this_kern,
                                                    np.zeros(
                                                        qfactor - len(this_kern) % qfactor,
                                                        dtype=np.int64
                                                    )
                                                )
                                            for i in range(qfactor):
                                                k |= ((this_kern[i::qfactor]
                                                       & (2**quantization[ll]-1))) \
                                                    << (i * quantization[ll])
                                        n += 1
                                        mask >>= 1
                                if debug:
                                    with np.printoptions(formatter={'int': '{0:02x}'.format}):
                                        print(f'Layer {ll} processor {p} channel '
                                              f'{ch + ie * in_expand_thresh[ll]} m[{m}..{m+n-1}] '
                                              f'of {output_chan[ll]}: {k}')

                                if flatten[ll]:
                                    for _, e in enumerate(k):
                                        col_target = add_kernel_data(ll, p, col_target, e)
                                else:
                                    for i in range(ksize):
                                        col_target = add_kernel_data(ll, p, col_target,
                                                                     k[ksize - i - 1])

                            else:  # When expanding, need to pad with zero kernels if needed
                                for _ in range(ksize // qfactor):
                                    col_target = add_kernel_data(ll, p, col_target, 0)

                        # Consume kernels
                        if not flatten[ll]:
                            col += qfactor
                            m += qfactor
                        else:
                            col += 1
                            m += 1
                    else:
                        m += qfactor

            if kern_offs[ll] + col_target < tc.dev.mask_width(p) \
               and kernels_used[p][kern_offs[ll] + col_target] > 0:  # Partials
                col_target += 1
            while col_target - start_col < kern_len[ll]:
                col_target = add_kernel_data(ll, p, col_target, 0)
            if flatten[ll]:
                kern_len[ll] = col_target
            else:
                assert kern_len[ll] == col_target - start_col
            proc_kern_max[p] = kern_offs[ll] + kern_len[ll]
            ch += 1
            m = 0

    if verbose:
        print('\nKernel map:')
        print_map(layers, kernel_map)

    if verify or not (embedded_code or mexpress):
        if verify:
            apb.output('int verify_kernels(void)\n{\n')
        # Write in-line
        for p in range(tc.dev.MAX_PROC):
            for col in range(0, tc.dev.mask_width(p)):
                ll = kernel_map[p][col]
                if ll != _INVALID_VALUE:
                    k = kernel_data[p][col]
                    apb.write_kern(ll, p, col, k, verify_only=verify, calcx4=calcx4)
        if verify:
            apb.output('  return 1;\n}\n\n')
    if embedded_code or mexpress:
        # Write kernels, combining layers and processors where possible to reduce the number
        # of constants and calls to memcpy.
        apb.output('// Kernels:\n')

        if not mexpress:
            for p in range(tc.dev.MAX_PROC):
                for col in range(0, tc.dev.mask_width(p)):
                    ll = kernel_map[p][col]
                    if ll != _INVALID_VALUE:
                        k = kernel_data[p][col]
                        offs = _WORDS_PER_KERNEL * col
                        kernel_values[p][offs] = k[0] & 0xff
                        kernel_values[p][offs + 1] = (k[1] & 0xff) << 24 \
                            | (k[2] & 0xff) << 16 | (k[3] & 0xff) << 8 | k[4] & 0xff
                        kernel_values[p][offs + 2] = (k[5] & 0xff) << 24 \
                            | (k[6] & 0xff) << 16 | (k[7] & 0xff) << 8 | k[8] & 0xff

            # First, define the weights (will move to header file)
            # Combining memcopy() requires stacked memories
            max_col = [-1] * tc.dev.MAX_PROC
            min_col = [tc.dev.MASK_WIDTH_LARGE if not legacy_kernels else 0] * tc.dev.MAX_PROC
            for p in range(0, tc.dev.MAX_PROC):
                for col in range(0, tc.dev.mask_width(p)):
                    ll = kernel_map[p][col]
                    if ll != _INVALID_VALUE:
                        max_col[p] = col
                        min_col[p] = min(min_col[p], col)
            p = 0
            while p < tc.dev.MAX_PROC:
                if max_col[p] >= 0:
                    start = p
                    while (
                            max_col[p] == tc.dev.MASK_OFFS and
                            p+1 < tc.dev.MAX_PROC and
                            max_col[p+1] >= 0 and
                            min_col[p+1] == 0 and
                            (start & ~(tc.dev.P_NUMPRO-1)) == (p+1 & ~(tc.dev.P_NUMPRO-1))
                    ):
                        p += 1
                    # Combine multiple channels into one define
                    k = None
                    for i in range(start, p + 1):
                        if k is None:
                            k = kernel_values[i][
                                min_col[i] * _WORDS_PER_KERNEL:
                                (max_col[i] + 1) * _WORDS_PER_KERNEL
                            ]
                        else:
                            k = np.concatenate(
                                (k, kernel_values[i][
                                        min_col[i] * _WORDS_PER_KERNEL:
                                        (max_col[i] + 1) * _WORDS_PER_KERNEL
                                    ])
                            )

                    apb.output_define(k, f'KERNELS_{start}', '0x%08x', 8)
                p += 1

            # Second, initialize static const variables as source for memcpy
            p = 0
            while p < tc.dev.MAX_PROC:
                if max_col[p] >= 0:
                    span = max_col[p] + 1 - min_col[p]
                    start = p
                    while (
                            max_col[p] == tc.dev.MASK_OFFS and
                            p+1 < tc.dev.MAX_PROC and
                            max_col[p+1] >= 0 and
                            min_col[p+1] == 0 and
                            (start & ~(tc.dev.P_NUMPRO-1)) == (p+1 & ~(tc.dev.P_NUMPRO-1))
                    ):
                        p += 1
                        span += max_col[p] + 1 - min_col[p]
                    if riscv_flash:
                        apb.output(rv.RISCV_FLASH)
                    apb.output(f'static const uint32_t kernels_{start}[] = KERNELS_{start};\n')
                p += 1
            apb.output('\n')

            # Generate code to load the weights using memcpy
            apb.output('void memcpy_96to128(uint32_t *dst, const uint32_t *src, int n)\n{\n')
            apb.output('  while (n-- > 0) {\n'
                       '    *dst++ = *src++;\n'
                       '    *dst++ = *src++;\n'
                       '    *dst++ = *src++;\n'
                       '    *dst++ = 0;  // Execute write\n'
                       '  }\n}\n\n')
        else:
            # When using the express loader, gather all consecutive kernels for each processor
            # and pack them.
            zero_kernel = np.array([0] * 9, dtype=np.uint8)
            k = None

            for p in range(tc.dev.MAX_PROC):
                # Find min/max from kernel_map
                max_col = -1
                min_col = tc.dev.mask_width(p) if not legacy_kernels else 0
                for col in range(0, tc.dev.mask_width(p)):
                    ll = kernel_map[p][col]
                    if ll != _INVALID_VALUE:
                        max_col = col
                        min_col = min(min_col, col)
                if max_col >= 0:
                    for col in range(min_col, max_col + 1):
                        ll = kernel_map[p][col]
                        if ll != _INVALID_VALUE:
                            new_k = (kernel_data[p][col] & 0xff).astype(np.uint8)
                        else:
                            new_k = zero_kernel
                        if k is None:
                            k = new_k
                        else:
                            k = np.concatenate((k, new_k))

                    # Round up to multiple of 4
                    if len(k) % 4 != 0:
                        k = np.concatenate((k, zero_kernel[:4 - len(k) % 4]))
                    # '>u4' swaps endianness to what the hardware needs, `view` packs into 32-bit
                    if not blocklevel:
                        apb.output_define(k.view(dtype='>u4'), f'KERNELS_{p}', '0x%08x', 8)
                    else:
                        addr = tc.dev.C_GROUP_OFFS * (p // tc.dev.P_NUMPRO) \
                            + tc.dev.C_MRAM_BASE + (p % tc.dev.P_NUMPRO) * tc.dev.MASK_OFFS * 16
                        apb.write(addr + min_col * 4 | 0x01, 0x01)
                        kb = k.view(dtype=">u4")
                        for _, e in enumerate(kb):
                            apb.write(addr, e)
                            addr += 4

                    if riscv_flash:
                        apb.output(rv.RISCV_FLASH)
                    apb.output(f'static const uint32_t kernels_{p}[] = KERNELS_{p};\n')
                    k = None
            apb.output('\n')

        if not blocklevel:
            apb.output('void load_kernels(void)\n{\n')
            max_col = [-1] * tc.dev.MAX_PROC
            min_col = [tc.dev.MASK_WIDTH_LARGE if not legacy_kernels else 0] * tc.dev.MAX_PROC
            for p in range(0, tc.dev.MAX_PROC):
                for col in range(0, tc.dev.mask_width(p)):
                    ll = kernel_map[p][col]
                    if ll != _INVALID_VALUE:
                        max_col[p] = col
                        min_col[p] = min(min_col[p], col)
            p = 0
            while p < tc.dev.MAX_PROC:
                if max_col[p] >= 0:
                    span = max_col[p] + 1 - min_col[p]
                    start = p
                    addr = apb.apb_base + tc.dev.C_GROUP_OFFS * (p // tc.dev.P_NUMPRO) \
                        + tc.dev.C_MRAM_BASE + (p % tc.dev.P_NUMPRO) * tc.dev.MASK_OFFS * 16
                    while (
                            max_col[p] == tc.dev.MASK_OFFS and
                            p+1 < tc.dev.MAX_PROC and
                            max_col[p+1] >= 0 and
                            min_col[p+1] == 0 and
                            (start & ~(tc.dev.P_NUMPRO-1)) == (p+1 & ~(tc.dev.P_NUMPRO-1))
                    ):
                        p += 1
                        span += max_col[p] + 1 - min_col[p]
                    assert addr % 16 == 0
                    if not mexpress:
                        apb.output('  memcpy_96to128((uint32_t *)'
                                   f' 0x{addr + min_col[start] * 16:08x},'
                                   f' kernels_{start}, {span});\n')
                    else:
                        apb.output('  *((volatile uint8_t *)'
                                   f' 0x{addr + min_col[start] * 4 | 0x01:08x}) = 0x01; '
                                   '// Set address\n')
                        apb.output(f'  memcpy32((uint32_t *) 0x{addr:08x}, '
                                   f'kernels_{start}, {(span * 9 + 3) // 4});\n')
                p += 1

            apb.output('}\n\n')

    return kern_offs, kern_len
