###################################################################################################
# Copyright (C) 2019-2023 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
###################################################################################################
"""
Kernel related functions
"""
import sys
from typing import Optional

import numpy as np

from . import console, op, rv, state
from . import tornadocnn as tc
from .eprint import eprint, eprint_noprefix, wprint
from .names import layer_pfx
from .utils import ffs, fls, popcount

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
    width = int(np.log10(layers)) + 1
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
            print_fn(f'{val:>{width}}', end='')
        print_fn('')
    print_fn('-' * tc.dev.MASK_WIDTH_LARGE * width)


def load(  # pylint: disable=too-many-branches,too-many-statements
        embedded_code,
        apb,
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
        conv_groups,
        flatten=None,
        verify=False,
        api=False,
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
    # Cache for faster access
    bypass = state.bypass
    calcx4 = state.calcx4
    debug = state.debug
    legacy_kernels = state.legacy_kernels
    mexpress = state.mexpress
    quad = state.fast_fifo_quad
    start_layer = state.first_layer_used
    start_offs = state.weight_start
    zero_sram = state.zero_sram
    riscv_flash = state.riscv_flash and not state.riscv_cache

    # Kernels: Stack kernels; write only the kernels needed
    proc_kern_max = np.zeros((tc.dev.MAX_PROC), dtype=np.int64)
    kern_offs = np.full((layers), fill_value=start_offs, dtype=np.int64)
    kern_len = np.zeros((layers), dtype=np.int64)
    kern_count = np.zeros((layers), dtype=np.int64)
    kern_ochan = np.zeros((layers), dtype=np.int64)
    kernel_map = np.full((tc.dev.MAX_PROC, tc.dev.MASK_WIDTH_LARGE),
                         fill_value=_INVALID_VALUE, dtype=np.int64)
    kernels_used = np.zeros((tc.dev.MAX_PROC, tc.dev.MASK_WIDTH_LARGE), dtype=np.int64)
    kernel_data = np.zeros((tc.dev.MAX_PROC, tc.dev.MASK_WIDTH_LARGE, 9), dtype=np.uint8)
    # There are four 32-bit words per 9-byte kernel.
    # The value map is initialized with zeros so we can later ignore unused entries and use
    # memcpy() on initialized and uninitialized data.
    kernel_values = np.zeros((tc.dev.MAX_PROC, tc.dev.MASK_WIDTH_LARGE * _WORDS_PER_KERNEL),
                             dtype=np.int64)
    if debug:
        print('\nLoading Kernels...')

    def check_kernel_mem(
            ll: int,
            p: int,
            offs: int,
            length: Optional[int] = None,
            error: bool = True,
            reverse: bool = False,
    ) -> bool:
        """Check that there is enough space at index `offs` for processor `p`"""
        assert tc.dev is not None
        if length is None:
            length = kern_len[ll]
        if reverse and offs < 0 or not reverse and offs + length > tc.dev.mask_width(p):
            if error:
                eprint(f'\n{layer_pfx(ll)}Kernel memory exhausted at processor {p}; '
                       f'offset: {offs} (0x{offs:04x}), needed: {length}.\n\n'
                       'Kernel map so far:', exit_code=None)
                print_map(layers, kernel_map, print_fn=eprint_noprefix)
                sys.exit(1)
            return False
        return True

    with console.Progress(start=True) as progress:
        task0 = progress.add_task(description='Arranging weights...', total=layers-start_layer)
        for ll in range(start_layer, layers):
            if operator[ll] == op.NONE or bypass[ll] or kernel[ll] is None:
                assert kern_len[ll] == 0
                assert kern_offs[ll] == start_offs
                progress.advance(task0)
                continue

            qfactor = 8 // abs(quantization[ll])

            if flatten[ll]:
                kernel_reshaped = kernel[ll].reshape(
                    output_chan[ll],
                    in_expand[ll],
                    -1,
                ).swapaxes(1, 2).reshape(
                    output_chan[ll] * in_expand_thresh[ll],
                    -1,
                    kernel_size[ll][0],
                    kernel_size[ll][1],
                )

                in_exp = 1
                in_chan = in_expand_thresh[ll]
            elif calcx4[ll]:
                # FIXME for output channels % 4 != 0
                assert output_chan[ll] % 4 == 0
                kernel_reshaped = kernel[ll].reshape(
                    output_chan[ll] // 4,
                    4,
                    -1,
                ).transpose(1, 0, 2).reshape(
                    kernel[ll].shape
                )

                in_exp = in_expand[ll]
                in_chan = input_chan[ll]
            elif ll == 0 and quad and qfactor != 1:
                # FIXME for output channels % (4 * qfactor) != 0
                assert output_chan[ll] % (4 * qfactor) == 0
                kernel_reshaped = kernel[ll].reshape(
                    output_chan[ll] // (4 * qfactor),
                    qfactor,
                    4,
                    input_chan[ll],
                    kernel_size[ll][0] * kernel_size[ll][1],
                ).transpose(0, 2, 1, 3, 4).reshape(
                    kernel[ll].shape
                )

                in_exp = in_expand[ll]
                in_chan = input_chan[ll]
            else:
                if operator[ll] == op.LINEAR:
                    kernel_reshaped = kernel[ll].reshape(
                        -1,
                        kernel_size[ll][0],
                        kernel_size[ll][1],
                    )
                else:
                    kernel_reshaped = kernel[ll]
                in_exp = in_expand[ll]
                in_chan = input_chan[ll]

            if quantization[ll] == -1:
                kernel_reshaped = kernel_reshaped.copy().clip(-1, 0)

            if np.ndim(kernel_reshaped) > 2:
                if kernel_reshaped.shape[-1] != kernel_size[ll][0] \
                   or kernel_reshaped.shape[-2] != kernel_size[ll][1]:
                    eprint(f'{layer_pfx(ll)}The configured kernel dimensions ({kernel_size[ll][0]}'
                           f'x{kernel_size[ll][1]}) do not match the weights file '
                           f'({kernel_reshaped.shape[-1]}x{kernel_reshaped.shape[-2]})!')
            else:
                if kernel_reshaped.shape[-1] != kernel_size[ll][0]:
                    eprint(f'{layer_pfx(ll)}The configured kernel dimensions ({kernel_size[ll][0]}'
                           f') do not match the weights file ({kernel_reshaped.shape[-1]})!')

            proc_map = processor_map[ll]
            if ll == 0 and quad:
                proc_map &= 2**tc.dev.P_NUMPRO - 1
            first_proc = ffs(proc_map)
            last_proc = fls(proc_map)
            ch = 0
            m = 0

            ksize = kernel_size[ll][0] * kernel_size[ll][1]
            next_layer_map = output_processor_map[ll]
            first_output_proc = ffs(next_layer_map)
            start_col = first_output_proc % tc.dev.P_SHARED  # First target column out of 4 shared
            if start_col > 0 and quantization[ll] != 8:
                wprint(f'{layer_pfx(ll)}Warning: {quantization[ll]}-bit quantization uses '
                       'unaligned output processors, this may cause issues')

            # Determine the number of kernels that need to be programmed. Since each instance
            # spans 4 processors, kernels for all instances that have a single processor enabled
            # need to be written, i.e. round down the first. The last does not need to be rounded
            # up because hardware takes care of it.
            # When using kernels smaller than 8 bit, round up to the next 8-bit boundary
            # Gaps are accounted for like any other kernel.

            # This extends the kernels to the right for output expansion
            if out_expand[ll] > 1:
                first_output_proc -= start_col

            # MAX7800X devices currently support only groups=1 and groups equal to input channels
            # equal to output channels.
            if conv_groups[ll] == 1:
                kc = (1 + fls(next_layer_map) - first_output_proc) \
                    * out_expand[ll] * in_exp
                kern_ochan[ll] = kern_count[ll] = kc + start_col * out_expand[ll] * in_exp
            else:
                kc = in_exp
                kern_count[ll] = kc + start_col * in_exp
                kern_ochan[ll] = (1 + fls(next_layer_map) - first_output_proc) \
                    * in_exp + start_col * in_exp

            if not legacy_kernels and flatten[ll]:
                kc *= kernel_reshaped.shape[1]
                kern_count[ll] *= kernel_reshaped.shape[1]  # FIXME
                kc -= (out_expand[ll] * popcount(next_layer_map) - output_chan[ll]) \
                    * kernel_reshaped.shape[1]
                kern_count[ll] -= (out_expand[ll] * popcount(next_layer_map) - output_chan[ll]) \
                    * kernel_reshaped.shape[1]
                kern_ochan[ll] = kern_count[ll]

            # Pack kernels to 72-bit words, while ensuring there is enough space when using 1/2/4
            # bit kernels where the kernel count requires padding.
            res = (kc % qfactor) * ksize * (qfactor - 1)
            kern_len[ll] = (kc * ksize * abs(quantization[ll]) + res + 71) // 72

            if ll == 0 and quad:
                kern_len[0] = (kern_len[0] + 3) // 4
                kern_count[0] = (kern_count[0] + 3) // 4
                kern_ochan[0] = (kern_ochan[0] + 3) // 4

            def search_kernel_mem(
                    ll: int,
                    offs: int,
                    first_proc: int,
                    last_proc: int,
                    proc_map: int,
                    reverse: bool = False,
                    error: bool = True,
            ) -> int:
                """Search kernel memory for free space"""
                assert tc.dev is not None

                def kernel_mem_mask(
                        offs: int,
                ):
                    """Return the mask bits for the kernel memory at offset `offs`"""
                    assert tc.dev is not None
                    if offs + 1 >= tc.dev.MASK_WIDTH_SMALL:
                        return tc.dev.MASK_INSTANCE_SMALL-1
                    return tc.dev.MASK_INSTANCE_LARGE-1

                p = first_proc
                # Find the first free column for the first processor
                while kernel_map[p][offs] != _INVALID_VALUE:
                    # Start at a multiple of 4 - round up to next multiple
                    if not reverse:
                        offs += tc.dev.P_SHARED
                    else:
                        offs -= tc.dev.P_SHARED
                    if not check_kernel_mem(ll, p, offs, error=error, reverse=reverse):
                        return -1

                while p < last_proc+1:
                    if (proc_map >> p) & 1 == 0:
                        # Skip unused processors
                        p += 1
                        continue

                    # For this processor, is there space for all kernels starting at
                    # column 'offs'?
                    if not reverse:
                        if tc.dev.REQUIRE_WEIGHT_MASK and conv_groups[ll] > 1:
                            # Ensure that all kernels for this layer are in the same memory
                            # instance
                            # Large or small memory instance mask?
                            mmask = kernel_mem_mask(offs + 1)
                            # Round up to next instance
                            if (offs + 1) & ~mmask != (offs + kern_len[ll]) & ~mmask:
                                offs = ((offs + mmask) & ~mmask) - 1
                            # Update mask in case we move into the small instances
                            mmask = kernel_mem_mask(offs + 1)
                            if (offs + 1) & ~mmask != (offs + kern_len[ll]) & ~mmask:
                                # Cannot ever make this fit since we just ran out of large
                                # instances
                                return -1
                        start_range = offs
                        end_range = offs + kern_len[ll]
                        step_range = 1
                    else:
                        # Note: reverse can only ever be true
                        # when kern_len[ll] <= MASK_INSTANCE_SMALL
                        if tc.dev.REQUIRE_WEIGHT_MASK and conv_groups[ll] > 1:
                            # Ensure that all kernels for this layer are in the same memory
                            # instance
                            # Large or small memory instance mask?
                            mmask = kernel_mem_mask(offs - 1)
                            if (offs - 1) & ~mmask != (offs - kern_len[ll]) & ~mmask:
                                offs &= ~mmask
                        start_range = offs
                        end_range = offs - kern_len[ll]
                        step_range = -1
                    for i in range(start_range, end_range, step_range):
                        if kernel_map[p][i] != _INVALID_VALUE:
                            # No, go to the next candidate
                            # (at least one more than what we're looking at, rounded)
                            if not reverse:
                                offs = (i + 1 + tc.dev.P_SHARED-1) & ~(tc.dev.P_SHARED-1)
                            else:
                                offs = (i - 1) & ~(tc.dev.P_SHARED-1)
                            if not check_kernel_mem(ll, p, offs, error=error):
                                return -1
                            # Reset to start at first processor again
                            p = first_proc - 1  # Subtract 1 since it's increased again below
                            break
                    # Check next processor
                    p += 1
                return offs

            # Find space for kernels
            if not state.greedy_kernel_allocator:
                for p in range(first_proc, last_proc+1):
                    if (proc_map >> p) & 1 == 0:
                        # Unused processor
                        continue
                    # Get highest offset for all used processors
                    kern_offs[ll] = max(proc_kern_max[p], kern_offs[ll])
            else:
                # Find the first block of kern_len[ll] size that is available for all used
                # processors.
                # Initially, start looking at 0 and subsequently at the first available for the
                # previously examined processors. Stop looking at the highest used offset for all
                # processors.

                search_col = -1  # Nothing found
                if state.new_kernel_loader and not (ll > 0 and calcx4[ll] and not calcx4[ll-1]):
                    # Check whether all processors used have extended mask space
                    extended_masks = tc.dev.MASK_WIDTH_LARGE > tc.dev.MASK_WIDTH_SMALL  # 2 sizes?
                    if extended_masks:
                        for p in range(first_proc, last_proc+1):
                            # Any used processors does not have the extended space
                            if (proc_map >> p) & 1 == 1 and not tc.dev.mask_large(p):
                                extended_masks = False
                                break

                    # Try the end of kernel memory first for processors with extended memory
                    if extended_masks and (not tc.dev.REQUIRE_WEIGHT_MASK
                                           or conv_groups[ll] == 1
                                           or kern_len[ll] <= tc.dev.MASK_INSTANCE_SMALL):
                        search_col = search_kernel_mem(ll, tc.dev.MASK_WIDTH_LARGE - 1,
                                                       first_proc, last_proc, proc_map,
                                                       reverse=True, error=False)

                # If nothing found at the end of kernel memory, start search at the beginning
                if search_col == -1:
                    search_col = (start_offs + tc.dev.P_SHARED - 1) & ~(tc.dev.P_SHARED - 1)
                    search_col = search_kernel_mem(ll, search_col, first_proc, last_proc, proc_map)

                    # All used processors have kernel_len space starting at this column
                    kern_offs[ll] = search_col

                    if ll > 0 and calcx4[ll] and not calcx4[ll-1]:
                        # FIXME: This is a quick workaround that should be properly addressed for
                        # mixed non-x4/x4 situations (most common: quad-fast-fifo input and calcx4
                        # in the rest of the network)
                        kern_offs[ll] *= 4

                    # We don't have to use dummy columns if there's space available on the left
                    if not state.new_kernel_loader:
                        kern_offs[ll] = \
                            max(0, kern_offs[ll] - (((ffs(next_layer_map) % tc.dev.P_SHARED)
                                                    + qfactor - 1) // qfactor))

                    kern_offs[ll] = (kern_offs[ll] + tc.dev.P_SHARED-1) & ~(tc.dev.P_SHARED-1)
                else:
                    search_col -= kern_len[ll] - 1
                    # All used processors have kernel_len space starting at this column
                    kern_offs[ll] = search_col
                    kern_offs[ll] = kern_offs[ll] & ~(tc.dev.P_SHARED-1)

            # Check for overflow
            check_kernel_mem(ll, last_proc, kern_offs[ll])

            proc_mask = 2**qfactor - 1

            # Start at the first used instance
            this_map_init = next_layer_map >> ffs(next_layer_map)

            def add_kernel_data(ll, p, col_target, b):
                ct = col_target
                if ll == 0 and quad:
                    ct //= 4
                    p += col_target % 4 * tc.dev.P_NUMPRO
                col = kern_offs[ll] + ct
                check_kernel_mem(ll, p, col, length=1)

                if kernels_used[p][col] == 0:  # Update kernel map
                    assert kernel_map[p][col] == _INVALID_VALUE
                    kernel_map[p][col] = ll

                assert kernels_used[p][col] <= 8
                assert isinstance(b, np.int64), f'Kernel is type {type(b)} instead of numpy.int64'
                assert 0 <= b <= 255, f'Trying to add kernel value {b}'
                kernel_data[p][col][8 - kernels_used[p][col]] = b & 0xff
                kernels_used[p][col] += 1

                if kernels_used[p][col] == 9:  # Flush
                    col_target += 1  # Write 1

                return col_target

            task1 = progress.add_task(f'Layer {ll}...', total=1+last_proc-first_proc)
            for p in range(first_proc, last_proc + 1):
                if (proc_map >> p) & 1 == 0:
                    # Unused source processor
                    progress.advance(task1)
                    continue
                # Skip start_col processors. Each takes up ksize bytes, or ksize // 9 full
                # kernel words. There are col_bytes leftover bytes.
                col_target, col_bytes = divmod(start_col * ksize * in_exp, 9)
                # Pad out the leftovers
                for _ in range(col_bytes // qfactor):  # FIXME for quantization
                    col_target = add_kernel_data(ll, p, col_target, np.int64(0))

                out_range = out_expand[ll] if conv_groups[ll] == 1 else 1
                for expand in range(out_range):
                    this_map = this_map_init
                    if conv_groups[ll] == 1:
                        col = expand * out_expand_thresh[ll]
                        stop_col = col + out_expand_thresh[ll]
                    else:
                        col = expand
                        stop_col = expand + 1

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

                        in_ch = in_chan
                        if flatten[ll]:
                            in_ch *= qfactor
                        src_offs = ch + m * in_ch

                        for ie in range(in_exp):
                            mask = this_mask

                            n = 0
                            if ie * in_expand_thresh[ll] + ch < in_ch \
                               and src_offs < len(kernel_reshaped):
                                if not flatten[ll]:
                                    k = np.zeros_like(kernel_reshaped[src_offs].reshape(-1))
                                else:
                                    k = np.empty((0), dtype=np.int64)
                                for i in range(qfactor):
                                    if m < output_chan[ll]:
                                        # Cycle through phases
                                        idx = n + ie * qfactor
                                        koffs = src_offs + (idx % in_exp) * in_expand_thresh[ll] \
                                            + (idx // in_exp) * in_chan
                                        if koffs < len(kernel_reshaped):
                                            this_kern = kernel_reshaped[koffs].reshape(-1) \
                                                & (2**abs(quantization[ll])-1)
                                            if not flatten[ll]:
                                                k |= this_kern << (i * abs(quantization[ll]))
                                            elif len(k) > 0:
                                                k = np.append(k, this_kern)
                                            else:
                                                k = this_kern
                                        n += 1
                                    mask >>= 1
                                if debug:
                                    with np.printoptions(formatter={'int': '{0:02x}'.format}):
                                        print(f'{layer_pfx(ll)}Processor {p} channel '
                                              f'{ch + ie * in_expand_thresh[ll]} m[{m}..{m+n-1}] '
                                              f'of {output_chan[ll]}: {k}')
                                if flatten[ll]:
                                    if len(k) % qfactor != 0:
                                        k = np.append(
                                            k,
                                            np.zeros(
                                                qfactor - len(k) % qfactor,
                                                dtype=np.int64,
                                            ),
                                        )
                                    for i in range(0, len(k) // qfactor):
                                        e = k[i * qfactor]
                                        for j in range(1, qfactor):
                                            e |= k[i * qfactor + j] << (j * abs(quantization[ll]))
                                        col_target = add_kernel_data(ll, p, col_target, e)
                                else:
                                    for i in range(ksize):
                                        col_target = add_kernel_data(ll, p, col_target,
                                                                     k[ksize - i - 1])

                            else:  # When expanding, need to pad with zero kernels if needed
                                for _ in range(ksize // qfactor):
                                    col_target = add_kernel_data(ll, p, col_target, np.int64(0))

                        # Consume kernels
                        if not flatten[ll]:
                            col += qfactor
                            m += qfactor
                        else:
                            col += 1
                            m += 1

                if ll == 0 and quad:
                    col_target = (col_target - start_col + 3) // 4 + start_col
                if kern_offs[ll] + col_target < tc.dev.mask_width(p) \
                   and kernels_used[p][kern_offs[ll] + col_target] > 0:  # Partials
                    col_target += 1
                while col_target - start_col < kern_len[ll]:
                    col_target = add_kernel_data(ll, p, col_target, np.int64(0))
                if flatten[ll]:
                    kern_len[ll] = col_target
                elif not state.new_kernel_loader:
                    kern_len[ll] = col_target - start_col
                proc_kern_max[p] = kern_offs[ll] + kern_len[ll]
                if ll == 0 and quad:
                    proc_kern_max[p + tc.dev.P_NUMPRO] = \
                        proc_kern_max[p + 2 * tc.dev.P_NUMPRO] = \
                        proc_kern_max[p + 3 * tc.dev.P_NUMPRO] = proc_kern_max[p]
                ch += 1
                m = 0
                progress.advance(task1)
            progress.remove_task(task1)
            progress.advance(task0)

    with console.Progress(start=True) as progress:
        if state.verbose:
            print('\nKernel map:')
            print_map(layers, kernel_map)

        if state.new_kernel_loader and not state.rtl_preload_weights:
            apb.output('static const uint32_t kernels[] = KERNELS;\n\n', api)

        if verify:
            if state.new_kernel_loader:
                apb.function_header(prefix='', function='mexpress_byte',
                                    return_type='static uint8_t',
                                    arguments='const uint32_t **addr, uint32_t *val, int *avail')
                apb.output(
                    '  if (*avail == 0) {\n'
                    '    *val = *(*addr)++;\n'
                    '    *avail = 4;\n'
                    '  }\n',
                    api,
                )
                # mexpress_byte
                apb.function_footer(return_value='(*val >> (--(*avail) * 8)) & 0xff')
            apb.function_header(function='verify_weights')
            if state.new_kernel_loader:
                apb.output(
                    '  uint32_t len, data, val;\n'
                    '  volatile uint32_t *addr, *ptr;\n'
                    '  const uint32_t *compare = kernels;\n'
                    '  int av;\n\n'
                    '  while ((addr = (volatile uint32_t *) *compare++) != 0) {\n'
                    '    len = (*compare++ * 4) / 9;\n'
                    '    ptr = (volatile uint32_t *)(((uint32_t)addr '
                    f'& 0x{~(tc.dev.MASK_OFFS - 1) & 0xffffffff:08x}) | '
                    f'(((uint32_t)addr & 0x{tc.dev.MASK_OFFS - 1:04x}) << 2));\n'
                    '    av = 0;\n'
                    '    val = 0;\n\n'
                    '    while (len-- > 0) {\n'
                    '      data = mexpress_byte(&compare, &val, &av);\n'
                    '      if (*ptr++ != data) {\n'
                    '        printf("Addr[0]: %08x, read: %08x, expected: %08x\\n", ptr-1, '
                    '*(ptr-1), data);\n'
                    '        return CNN_FAIL;\n'
                    '      }\n'
                    '      data = mexpress_byte(&compare, &val, &av) << 24 | '
                    'mexpress_byte(&compare, &val, &av) << 16\n'
                    '           | mexpress_byte(&compare, &val, &av) << 8 | '
                    'mexpress_byte(&compare, &val, &av);\n'
                    '      if (*ptr++ != data) {\n'
                    '        printf("Addr[1]: %08x, read: %08x, expected: %08x\\n", ptr-1, '
                    '*(ptr-1), data);\n'
                    '        return CNN_FAIL;\n'
                    '      }\n'
                    '      data = mexpress_byte(&compare, &val, &av) << 24 | '
                    'mexpress_byte(&compare, &val, &av) << 16\n'
                    '           | mexpress_byte(&compare, &val, &av) << 8 | '
                    'mexpress_byte(&compare, &val, &av);\n'
                    '      if (*ptr++ != data) {\n'
                    '        printf("Addr[2]: %08x, read: %08x, expected: %08x\\n", ptr-1, '
                    '*(ptr-1), data);\n'
                    '        return CNN_FAIL;\n'
                    '      }\n'
                    '      if (*ptr++ != 0) {\n'
                    '        printf("Addr[3]: %08x, read: %08x, expected: 00000000\\n", ptr-1, '
                    '*(ptr-1));\n'
                    '        return CNN_FAIL;\n'
                    '      }\n'
                    '    }\n'
                    '  }\n',
                    api,
                )
            else:
                # Write in-line
                for p in range(tc.dev.MAX_PROC):
                    for col in range(0, tc.dev.mask_width(p)):
                        ll = kernel_map[p][col]
                        if ll != _INVALID_VALUE:
                            apb.write_kern(ll, p, col, kernel_data[p][col],
                                           verify_only=verify, calc_x4=calcx4[ll],
                                           kern_offs=kern_offs,
                                           count=in_expand[ll] * output_chan[ll] * 9
                                           * abs(quantization[ll])
                                           // (kernel_size[ll][0] * kernel_size[ll][1] * 8))
            apb.function_footer()  # verify_weights()

        if state.new_kernel_loader or not (embedded_code or mexpress) or any(calcx4):
            apb.function_header(function='load_weights')
            # Write (or store) in-line
            task = progress.add_task('Storing weights...  ', total=tc.dev.MAX_PROC)
            for p in range(tc.dev.MAX_PROC):
                for col in range(0, tc.dev.mask_width(p)):
                    ll = kernel_map[p][col]
                    if ll != _INVALID_VALUE:
                        k = kernel_data[p][col]
                        if not zero_sram or np.any(k != 0):
                            apb.write_kern(ll, p, col, k, calc_x4=calcx4[ll],
                                           kern_offs=kern_offs,
                                           count=in_expand[ll] * output_chan[ll] * 9
                                           * abs(quantization[ll])
                                           // (kernel_size[ll][0] * kernel_size[ll][1] * 8))
                progress.advance(task)

            if state.new_kernel_loader and not state.rtl_preload_weights:
                apb.output('  uint32_t len;\n'
                           '  volatile uint32_t *addr;\n'
                           '  const uint32_t *ptr = kernels;\n'
                           '\n'
                           '  while ((addr = (volatile uint32_t *) *ptr++) != 0) {\n',
                           api)
                if state.mexpress:
                    apb.output('    *((volatile uint8_t *) ((uint32_t) addr | 1)) = 0x01; '
                               '// Set address\n',
                               api)
                apb.output('    len = *ptr++;\n'
                           '    while (len-- > 0)\n'
                           '      *addr++ = *ptr++;\n'
                           '  }\n',
                           api)
            apb.function_footer()  # load_weights()

        else:  # embedded_code or mexpress
            # Write kernels, combining layers and processors where possible to reduce the number
            # of constants and calls to memcpy.
            apb.output('// Kernels:\n', api)

            if not mexpress:
                task = progress.add_task('Storing weights...  ', total=tc.dev.MAX_PROC)
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
                    progress.advance(task)

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
                            apb.output(rv.RISCV_FLASH, api)
                        apb.output(f'static const uint32_t kernels_{start}[] = KERNELS_{start};\n',
                                   api)
                    p += 1
                apb.output('\n', api)

                # Generate code to load the weights using memcpy
                apb.function_header(prefix='', function='memcpy_96to128', return_type='void',
                                    arguments='uint32_t *dst, const uint32_t *src, int n')
                apb.output('  while (n-- > 0) {\n'
                           '    *dst++ = *src++;\n'
                           '    *dst++ = *src++;\n'
                           '    *dst++ = *src++;\n'
                           '    *dst++ = 0;  // Execute write\n'
                           '  }\n', api)
                apb.function_footer(return_value='void')  # memcpy_96to128()
            else:
                # When using the express loader, gather all consecutive kernels for each processor
                # and pack them.
                zero_kernel = np.array([0] * 9, dtype=np.uint8)
                k = None

                task = progress.add_task('Storing weights...  ', total=tc.dev.MAX_PROC)
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
                        # '>u4' swaps endianness to what the hardware needs,
                        # `view` packs into 32-bit
                        if not state.block_mode:
                            apb.output_define(k.view(dtype='>u4'), f'KERNELS_{p}', '0x%08x', 8)
                        else:
                            addr = tc.dev.C_GROUP_OFFS * (p // tc.dev.P_NUMPRO) \
                                + tc.dev.C_MRAM_BASE \
                                + (p % tc.dev.P_NUMPRO) * tc.dev.MASK_OFFS * 16
                            apb.write(addr + min_col * 4 | 0x01, 0x01)
                            kb = k.view(dtype=">u4")
                            for e in kb:
                                apb.write(addr, e)
                                addr += 4

                        if riscv_flash:
                            apb.output(rv.RISCV_FLASH, api)
                        apb.output(f'static const uint32_t kernels_{p}[] = KERNELS_{p};\n', api)
                        k = None
                    progress.advance(task)
                apb.output('\n', api)

            if not state.block_mode:
                apb.function_header(function='load_weights')
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
                        addr = state.apb_base + tc.dev.C_GROUP_OFFS * (p // tc.dev.P_NUMPRO) \
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
                                       f' kernels_{start}, {span});\n', api)
                        else:
                            apb.output('  *((volatile uint8_t *)'
                                       f' 0x{addr + min_col[start] * 4 | 0x01:08x}) = 0x01; '
                                       '// Set address\n', api)
                            apb.output(f'  memcpy32((uint32_t *) 0x{addr:08x}, '
                                       f'kernels_{start}, {(span * 9 + 3) // 4});\n', api)
                    p += 1

                apb.function_footer()  # load_weights()

    return kern_offs, kern_len, kern_count, kern_ochan


def calcx4_index(k):
    """
    Re-arranges a kernel offset `k` for calcx4 support.
    """
    if not tc.dev.SUPPORT_CALCX4:
        return k

    if k < tc.dev.MASK_WIDTH_SMALL:
        return (k % 4) * (tc.dev.MASK_WIDTH_SMALL // 4) + k // 4

    k -= tc.dev.MASK_WIDTH_SMALL
    k = (k % 4) * ((tc.dev.MASK_WIDTH_LARGE - tc.dev.MASK_WIDTH_SMALL) // 4) \
        + k // 4
    return k + tc.dev.MASK_WIDTH_SMALL
