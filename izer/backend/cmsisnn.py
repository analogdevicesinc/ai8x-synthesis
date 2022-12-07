###################################################################################################
# Copyright (C) 2019-2022 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
###################################################################################################
"""
Routines to generate software CNNs using Arm's CMSIS NN library
"""
import os
import sys
from typing import Any, List

import numpy as np

from izer import assets, op, state, stats, toplevel
from izer import tornadocnn as tc
from izer.eprint import eprint, wprint
from izer.simulate import (conv1d_layer, conv2d_layer, convtranspose2d_layer, eltwise_layer,
                           passthrough_layer, pooling_layer, show_data)

from . import backend


class Backend(backend.Backend):
    """
    Backend for creating a C code CMSIS-NN network
    """

    def create_net(self) -> str:  # pylint: disable=too-many-locals,too-many-branches
        """
        Chain multiple CNN layers, create and save input and output
        """

        # Cache state locally for faster access
        activation = state.activation
        auto_input_dim = state.auto_input_dim
        avg_pool_rounding = state.avg_pool_rounding
        base_directory = state.base_directory
        bias = state.bias
        c_filename = state.c_filename
        conv_groups = state.conv_groups
        data = state.data
        dilation = state.dilation
        eltwise = state.eltwise
        flatten = state.flatten
        in_sequences = state.in_sequences
        input_chan = state.input_channels
        input_dim = state.input_dim
        kernel = state.weights
        kernel_size = state.kernel_size
        layers = state.layers
        legacy_test = state.legacy_test
        log = state.log
        log_filename = state.log_filename
        operands = state.operands
        operator = state.operator
        output_chan = state.output_channels
        output_dim = state.output_dim
        output_shift = state.output_shift
        output_width = state.output_width
        padding = state.padding
        pool = state.pool
        pool_average = state.pool_average
        pool_first = state.pool_first
        pool_stride = state.pool_stride
        pooled_dim = state.pooled_dim
        prefix = state.prefix
        quantization = state.quantization
        sample_filename = state.sample_filename
        stride = state.stride
        weight_filename = state.weight_filename

        wprint('CMSIS-NN code generation is unsupported.')

        if output_width[-1] != 8 and operator[-1] != op.LINEAR:
            wprint('CMSIS-NN network generator does not currently support `output_width` that '
                   'is not 8 when not using Linear. Forcing to 8 bit.')  # FIXME: 32-bit output
            output_width[-1] = 8

        final_size = 7 if output_width[-1] == 8 else 31

        input_dim_str: List[str] = [''] * layers
        output_dim_str: List[str] = [''] * layers
        kernel_size_str: List[str] = [''] * layers
        pool_str: List[str] = [''] * layers
        padding_str: List[str] = [''] * layers
        pool_stride_str: List[str] = [''] * layers
        stride_str: List[str] = [''] * layers

        for ll in range(layers):
            if quantization[ll] is None:
                quantization[ll] = 8  # Set default
            elif quantization[ll] != 8:  # FIXME: Support quantization
                wprint('CMSIS-NN network generator does not currently support `quantization` '
                       '!= 8. Forcing to 8 bit.')

            if output_shift[ll] is None:
                output_shift[ll] = 0  # Set default

            if operator[ll] != op.CONV1D:
                input_dim_str[ll] = f'{input_dim[ll][0]}x{input_dim[ll][1]}'
                output_dim_str[ll] = f'{output_dim[ll][0]}x{output_dim[ll][1]}'
                kernel_size_str[ll] = f'{kernel_size[ll][0]}x{kernel_size[ll][1]}'
                pool_str[ll] = f'{pool[ll][0]}x{pool[ll][1]}' \
                    if pool[ll][0] > 1 or pool[ll][1] > 1 else '0x0'
                padding_str[ll] = f'{padding[ll][0]}/{padding[ll][1]}'
                pool_stride_str[ll] = f'{pool_stride[ll][0]}/{pool_stride[ll][1]}'
                stride_str[ll] = f'{stride[ll][0]}/{stride[ll][1]}'
            else:
                input_dim_str[ll] = f'{input_dim[ll][0]}'
                output_dim_str[ll] = f'{output_dim[ll][0]}'
                kernel_size_str[ll] = f'{kernel_size[ll][0]}'
                pool_str[ll] = f'{pool[ll][0]}' \
                    if pool[ll][0] > 1 or pool[ll][1] > 1 else '0'
                padding_str[ll] = f'{padding[ll][0]}'
                pool_stride_str[ll] = f'{pool_stride[ll][0]}'
                stride_str[ll] = f'{stride[ll][0]}'

            if input_chan[ll] % conv_groups[ll] != 0 or output_chan[ll] % conv_groups[ll] != 0:
                eprint(f'Layer {ll}: convolution groups {conv_groups[ll]} does not divide '
                       f'the input channels {input_chan[ll]} or output channels '
                       f'{output_chan[ll]}.')

        test_name = prefix
        print(f'{test_name}...')

        os.makedirs(os.path.join(base_directory, test_name), exist_ok=True)

        # Redirect stdout?
        if log:
            state.output_is_console = False
            sys.stdout = open(
                os.path.join(base_directory, test_name, log_filename),
                mode='w',
                encoding='utf-8',
            )
            print(f'{" ".join(str(x) for x in sys.argv)}')
            assert tc.dev is not None
            print(f'{tc.dev.partnum}\n')
            print(f'{test_name}')

        filename = c_filename + '.c'
        sampledata_header = \
            open(
                os.path.join(base_directory, test_name, sample_filename),
                mode='w',
                encoding='utf-8',
            )
        weight_header = \
            open(
                os.path.join(base_directory, test_name, weight_filename),
                mode='w',
                encoding='utf-8',
            )

        with open(
            os.path.join(base_directory, test_name, filename),
            mode='w',
            encoding='utf-8',
        ) as c_file:
            toplevel.copyright_header(c_file)

            c_file.write(f'// {test_name}\n'
                         '// This file was @generated by '
                         f'{" ".join(str(x) for x in sys.argv)}\n\n')

            toplevel.header(c_file, embedded_code=True)

            # Pre-define data memory loader.
            d = data.transpose((1, 2, 0)).flatten()  # CHW -> HWC
            toplevel.c_define(sampledata_header, d, 'INPUT_DATA', fmt='%d', columns=16, size=8)
            input_size = d.size
            c_file.write('static const q7_t input_data[] = INPUT_DATA;\n')
            c_file.write(f'static const q{output_width[-1]-1}_t output_data[] = OUTPUT_DATA; '
                         '// Last conv layer output\n')

            # Pre-define the kernels and bias values
            for ll in range(layers):
                # Rearrange kernels when emulating a fully connected network using 1x1 Conv2D
                # CMSIS data uses HWC, PyTorch uses CHW
                if operator[ll] != op.NONE:
                    if kernel_size[ll] == [1, 1] and input_dim[ll] == [1, 1]:
                        w = kernel[ll]. \
                            reshape((output_chan[ll],
                                    input_chan[ll] //
                                    (auto_input_dim[ll][0] * auto_input_dim[ll][1]),
                                    auto_input_dim[ll][0], auto_input_dim[ll][1],
                                    kernel_size[ll][0], kernel_size[ll][1])). \
                            transpose((0, 4, 5, 2, 3, 1)). \
                            flatten()
                    elif flatten[ll]:
                        w = kernel[ll]. \
                            reshape((output_chan[ll],
                                    input_chan[ll],
                                    auto_input_dim[ll][0], auto_input_dim[ll][1],
                                    kernel_size[ll][0], kernel_size[ll][1])). \
                            transpose((0, 4, 5, 2, 3, 1)). \
                            flatten()
                    else:
                        w = kernel[ll]. \
                            reshape((output_chan[ll], input_chan[ll],
                                    kernel_size[ll][0], kernel_size[ll][1])). \
                            transpose((0, 2, 3, 1)). \
                            flatten()
                    toplevel.c_define(weight_header, w, f'WEIGHTS_{ll}', fmt='%d', columns=16,
                                      size=8)
                    if bias[ll] is not None:
                        b = bias[ll].flatten()
                    else:
                        # We need empty bias values (the Arm code needs them both for rounding of
                        # the shifted output, and it does not like NULL bias pointers)
                        b = np.zeros(output_chan[ll], dtype=np.int64)
                    toplevel.c_define(weight_header, b, f'BIAS_{ll}', fmt='%d', columns=16, size=8)
            c_file.write('\n')

            for ll in range(layers):
                if operator[ll] != op.NONE:
                    c_file.write(f'static const q7_t weights_{ll}[] = WEIGHTS_{ll};\n')
                    c_file.write(f'static const q7_t bias_{ll}[] = BIAS_{ll};\n')
            c_file.write('\n')

            # Compute buffer sizes
            col_buffer_size = 0
            img_buffer_size = 0
            for ll in range(layers):
                col_buffer_size = max(col_buffer_size,
                                      2*input_chan[ll]*kernel_size[ll][0]*kernel_size[ll][1])
                if pool[ll][0] > 1 or pool[ll][1] > 1:
                    # q15_t doesn't need 2*
                    col_buffer_size = max(col_buffer_size,
                                          pooled_dim[ll][0]*input_chan[ll])
                img_buffer_size = max(img_buffer_size,
                                      input_chan[ll]*input_dim[ll][0]*input_dim[ll][1],
                                      output_chan[ll]*output_dim[ll][0]*output_dim[ll][1])

            c_file.write(f'static q7_t buffer0[{max(img_buffer_size, input_size)}];\n')
            c_file.write(f'static q7_t buffer1[{img_buffer_size}];\n')
            c_file.write(f'static q15_t col_buffer[{col_buffer_size}];\n\n')

            c_file.write(f'int cnn_run(const q7_t *input, int input_size, '
                         f'q{final_size}_t **output, int *output_size)\n{{\n')

            # Compute layer-by-layer output and chain results into input
            buffer0, buffer1 = 'buffer0', 'buffer1'

            def run_eltwise(
                    data,
                    ll,
            ):
                """
                In-flight element-wise operations
                """
                if operator[ll] == op.NONE:
                    # Let element-wise do 32-bit, else 8-bit only
                    o_width = output_width[ll]
                else:
                    o_width = 8
                d_shape = data.shape

                data, out_size = eltwise_layer(
                    eltwise[ll],
                    ll,
                    data[0].shape,
                    output_shift[ll],
                    data,
                    output_width=o_width,
                    operands=operands[ll],
                )
                assert out_size[0] == d_shape[1] \
                    and out_size[1] == d_shape[2] and out_size[2] == d_shape[3]

                return data

            data_buf: List[Any] = [data]
            # Compute layer-by-layer output and chain results into input
            for ll in range(layers):
                # Concatenate input data if needed
                if in_sequences[ll] is not None:
                    if isinstance(in_sequences[ll], list):
                        try:
                            data = np.concatenate([data_buf[i + 1] for i in in_sequences[ll]],
                                                  axis=0)
                        except ValueError as err:
                            eprint('Error in input data concatenation layer:', err)
                    else:
                        data = data_buf[in_sequences[ll] + 1]
                else:
                    data = data_buf[-1]

                # Split data into multiple inputs if needed
                if operands[ll] > 1:
                    if ll == 0 and legacy_test:
                        data = np.array(np.split(data, operands[ll], axis=0))
                    elif legacy_test:
                        d = np.empty((operands[ll],
                                     data.shape[0], data.shape[1], data.shape[2] // operands[ll]),
                                     dtype=np.int64)
                        for i in range(operands[ll]):
                            d[i, :, :, :] = data[:, :, i::operands[ll]]
                        data = d
                    else:
                        data = np.array(np.split(data, operands[ll], axis=0))
                else:
                    data = np.expand_dims(data, 0)

                show_data(
                    ll,
                    data.shape,
                    data,
                    expand=1,
                    expand_thresh=1,
                    operation=operator[ll],
                    operands=operands[ll],
                )

                in_chan = input_chan[ll]

                # Run in-flight element-wise operations first?
                if operands[ll] > 1 and not pool_first[ll]:
                    eprint("Element-wise operations are currently not implemented for CMSIS-NN")
                    # FIXME: Support element-wise operations
                    data = np.expand_dims(run_eltwise(data, ll), 0)

                # Allow 1D <-> 2D and 2D W/L conversions
                if operator[ll] == op.CONV1D:
                    assert input_dim[ll][1] == 1
                    data = data.reshape(data.shape[0], data.shape[1], input_dim[ll][0])
                else:
                    data = data.reshape(data.shape[0], data.shape[1],
                                        input_dim[ll][0], input_dim[ll][1])

                # In-flight pooling
                data, out_size = pooling_layer(
                    ll,
                    data[0].shape,
                    pool[ll],
                    pool_stride[ll],
                    pool_average[ll],
                    data,
                    expand=1,
                    expand_thresh=1,
                    operation=operator[ll],
                    operands=data.shape[0],
                    rounding=avg_pool_rounding,
                    debug_data=None,
                )

                if operator[ll] == op.CONV1D:
                    assert out_size[0] == in_chan \
                        and out_size[1] == pooled_dim[ll][0] \
                        and pooled_dim[ll][1] == 1
                else:
                    assert out_size[0] == in_chan \
                        and out_size[1] == pooled_dim[ll][0] \
                        and out_size[2] == pooled_dim[ll][1]

                if operands[ll] > 1 and pool_first[ll]:
                    data = run_eltwise(data, ll)
                else:
                    data = np.squeeze(data, axis=0)

                # Convolution or passthrough
                if operator[ll] in [op.CONV2D, op.LINEAR]:
                    if flatten[ll]:
                        in_chan *= input_dim[ll][0] * input_dim[ll][1]
                        data = data.reshape(in_chan, 1, 1)
                        if state.verbose:
                            print(f"FLATTEN TO {in_chan}x1x1...\n")

                    out_buf, out_size = conv2d_layer(
                        ll,
                        data.shape,
                        kernel_size[ll],
                        output_shift[ll],
                        output_chan[ll],
                        padding[ll],
                        dilation[ll],
                        stride[ll],
                        activation[ll],
                        kernel[ll].reshape(
                            output_chan[ll],
                            in_chan,
                            kernel_size[ll][0],
                            kernel_size[ll][1]
                        ),
                        bias[ll],
                        data,
                        output_width=output_width[ll],
                        groups=conv_groups[ll],
                    )
                elif operator[ll] == op.CONVTRANSPOSE2D:
                    out_buf, out_size = convtranspose2d_layer(
                        ll,
                        data.shape,
                        kernel_size[ll],
                        output_shift[ll],
                        output_chan[ll],
                        padding[ll],
                        dilation[ll],
                        stride[ll],
                        [1, 1],  # output_padding
                        activation[ll],
                        kernel[ll].reshape(
                            output_chan[ll],
                            in_chan,
                            kernel_size[ll][0],
                            kernel_size[ll][1],
                        ),
                        bias[ll],
                        data,
                        output_width=output_width[ll],
                        groups=conv_groups[ll],
                    )
                elif operator[ll] == op.CONV1D:
                    out_buf, out_size = conv1d_layer(
                        ll,
                        data.shape,
                        kernel_size[ll][0],
                        output_shift[ll],
                        output_chan[ll],
                        padding[ll][0],
                        dilation[ll][0],
                        stride[ll][0],
                        activation[ll],
                        kernel[ll].reshape(
                            output_chan[ll],
                            input_chan[ll],
                            kernel_size[ll][0],
                        ),
                        bias[ll],
                        data,
                        output_width=output_width[ll],
                        groups=conv_groups[ll],
                    )
                elif operator[ll] == op.NONE:  # '0'D (pooling only or passthrough)
                    out_buf, out_size = passthrough_layer(
                        ll,
                        data.shape,
                        data,
                    )
                else:
                    eprint(f'Unknown operator `{op.string(operator[ll])}`.')

                assert out_size[0] == output_chan[ll] \
                    and out_size[1] == output_dim[ll][0] and out_size[2] == output_dim[ll][1]

                c_file.write(f'  // Layer {ll}: '
                             f'{str(operands[ll])+"x" if operands[ll] > 1 else ""}'
                             f'{input_chan[ll]}x{input_dim_str[ll]}'
                             f'{" flattened, " if flatten[ll] else ", "}')
                if pool[ll][0] > 1 or pool[ll][1] > 1:
                    c_file.write(f'{pool_str[ll]} {"avg" if pool_average[ll] else "max"} '
                                 f'pool with stride {pool_stride_str[ll]}')
                else:
                    c_file.write('no pooling')
                if operator[ll] in [op.CONV1D, op.CONV2D, op.CONVTRANSPOSE2D, op.LINEAR]:
                    conv_str = f', {op.string(operator[ll])} with kernel size ' \
                               f'{kernel_size_str[ll]}, ' \
                               f'stride {stride_str[ll]}, ' \
                               f'pad {padding_str[ll]}, ' \
                               f'{op.act_string(activation[ll])}, '
                else:
                    conv_str = ', no convolution, '
                c_file.write(conv_str +
                             f'{output_chan[ll]}x{output_dim_str[ll]} output\n')

                c_file.write(f'  // Dimensions: [{input_chan[ll]}, {input_dim[ll][0]}, '
                             f'{input_dim[ll][1]}]')
                if pool[ll][0] > 1 or pool[ll][1] > 1:
                    c_file.write(f' -> [{input_chan[ll]}, {pooled_dim[ll][0]}, '
                                 f'{pooled_dim[ll][1]}]')
                if flatten[ll]:
                    c_file.write(f' -> [{input_chan[ll]*pooled_dim[ll][0]*pooled_dim[ll][1]}, '
                                 '1, 1]')
                if operator[ll] != op.NONE:
                    c_file.write(f' -> {out_size}\n')
                else:
                    c_file.write('\n')

                source = 'input_data' if ll == 0 else buffer0

                if pool[ll][0] > 1 or pool[ll][1] > 1:
                    if ll == 0:
                        c_file.write('  memcpy(buffer0, input, input_size);'
                                     ' // Pooling may destroy input\n')
                    pool_type = 'ave' if pool_average[ll] else 'max'
                    if pool[ll][0] != pool[ll][1]:
                        c_file.write(f'  arm_{pool_type}pool_nonsquare_q7_HWC_nonsquare('
                                     f'{buffer0}, '
                                     f'{input_dim[ll][1]}, {input_dim[ll][0]}, '
                                     f'{input_chan[ll]}, {pool[ll][1]}, {pool[ll][0]}, 0, 0, '
                                     f'{pool_stride[ll][1]}, {pool_stride[ll][0]}, '
                                     f'{pooled_dim[ll][1]}, {pooled_dim[ll][0]}, '
                                     f'(q7_t *) col_buffer, {buffer1});\n')
                    else:
                        if input_dim[ll][0] == input_dim[ll][1]:
                            c_file.write(f'  arm_{pool_type}pool_q7_HWC({buffer0}, '
                                         f'{input_dim[ll][0]}, {input_chan[ll]}, '
                                         f'{pool[ll][0]}, 0, {pool_stride[ll][0]}, '
                                         f'{pooled_dim[ll][0]}, (q7_t *) col_buffer, '
                                         f'{buffer1});\n')
                        else:
                            c_file.write(f'  arm_{pool_type}pool_q7_HWC_nonsquare({buffer0}, '
                                         f'{input_dim[ll][1]}, {input_dim[ll][0]}, '
                                         f'{input_chan[ll]}, {pool[ll][0]}, 0, '
                                         f'{pool_stride[ll][0]}, '
                                         f'{pooled_dim[ll][1]}, {pooled_dim[ll][0]}, '
                                         f'(q7_t *) col_buffer, {buffer1});\n')
                    source = buffer1
                    buffer0, buffer1 = buffer1, buffer0

                if operator[ll] != op.NONE:
                    in_chan = input_chan[ll]
                    in_dim = pooled_dim[ll]
                    if flatten[ll]:
                        in_chan *= pooled_dim[ll][0] * pooled_dim[ll][1]
                        in_dim = [1, 1]

                    if operator[ll] in [op.CONVTRANSPOSE2D]:  # FIXME: Support ConvTranspose2d
                        eprint("CMSIS-NN generator does not currently support the operator "
                               f"`{op.string(operator[ll])}` in layer {ll}")

                    # FIXME: First check that everything is [-128, +127] and use s8
                    # function otherwise

                    # Check for squareness
                    if kernel_size[ll][0] == kernel_size[ll][1] \
                       and in_dim[0] == in_dim[1] \
                       and output_dim[ll][0] == output_dim[ll][1] \
                       and padding[ll][0] == padding[ll][1] \
                       and stride[ll][0] == stride[ll][1]:
                        # Detect fully connected layers
                        if operator[ll] == op.LINEAR:
                            assert in_dim == [1, 1] and output_dim[ll] == [1, 1]
                            if output_width[ll] == 8:
                                fn = 'q7'
                                shift = 7 - output_shift[ll]
                                cast = ''
                            else:
                                fn = 'q7_q31'
                                shift = 0
                                cast = '(q31_t *) '
                            c_file.write(f'  arm_fully_connected_{fn}({source}, '
                                         f'weights_{ll}, {in_chan}, {output_chan[ll]}, 7, '
                                         f'{shift}, bias_{ll}, {cast}{buffer1}, '
                                         'col_buffer);\n')
                        else:
                            fn = 'fast' if in_chan % 4 == 0 and output_chan[ll] % 2 == 0 \
                                else 'basic'
                            c_file.write(f'  arm_convolve_HWC_q7_{fn}({source}, '
                                         f'{in_dim[0]}, '
                                         f'{in_chan}, weights_{ll}, {output_chan[ll]}, '
                                         f'{kernel_size[ll][0]}, '
                                         f'{padding[ll][0]}, '
                                         f'{stride[ll][0]}, '
                                         f'bias_{ll}, 7,  {7 - output_shift[ll]}, {buffer1}, '
                                         f'{output_dim[ll][0]}, '
                                         'col_buffer, NULL);\n')
                    else:
                        c_file.write(f'  arm_convolve_HWC_q7_basic_nonsquare({source}, '
                                     f'{in_dim[1]}, {in_dim[0]}, '
                                     f'{in_chan}, weights_{ll}, {output_chan[ll]}, '
                                     f'{kernel_size[ll][1]}, {kernel_size[ll][0]}, '
                                     f'{padding[ll][1]}, {padding[ll][0]}, '
                                     f'{stride[ll][1]}, {stride[ll][0]},\n'
                                     '                                      '
                                     f'bias_{ll}, 7, {7 - output_shift[ll]}, {buffer1}, '
                                     f'{output_dim[ll][1]}, {output_dim[ll][0]}, '
                                     'col_buffer, NULL);\n')

                    assert out_size[0] == output_chan[ll] \
                        and out_size[1] == output_dim[ll][0] and out_size[2] == output_dim[ll][1]

                    if activation[ll] == op.ACT_RELU:
                        size = output_dim[ll][0] * output_dim[ll][1] * output_chan[ll]
                        if size < 65536:
                            c_file.write(f'  arm_relu_q7({buffer1}, {size});\n')
                        else:
                            c_file.write(f'  arm_relu32_q7({buffer1}, {size});\n')
                    elif activation[ll] is not None:  # FIXME: Support abs() activation
                        eprint("CMSIS-NN generator implements ReLU only.")
                    buffer0, buffer1 = buffer1, buffer0

                if not np.any(out_buf):
                    wprint(f'Layer {ll}: All output values for the given sample input are zero. '
                           'The generated known-answer test for this network may not be '
                           'meaningful. See the log file for details.')

                data_buf.append(out_buf.reshape(out_size))
                c_file.write('\n')
                data_cmsis = data_buf[-1].transpose((1, 2, 0)).flatten()
                if state.verbose:
                    print('TRANSPOSED (HWC) AND FLATTENED:')
                    print(data_cmsis)
                    print('')

            data = data_buf[-1]

            c_file.write(f'  *output = {"" if output_width[ll] == 8 else "(q31_t *) "}{buffer0};\n'
                         f'  *output_size = {data_cmsis.size};\n\n'
                         '  return 1;\n}\n\n')

            c_file.write('int main(void)\n{\n'
                         '  int i;\n'
                         f'  q{final_size}_t *output;\n'
                         '  int output_size;\n\n'
                         f'  cnn_run(input_data, {input_size}, &output, &output_size);\n\n')

            toplevel.c_define(sampledata_header, data_cmsis, 'OUTPUT_DATA', fmt='%d', columns=16,
                              size=8)
            c_file.write('  if (memcmp(output_data, output, output_size) == 0)\n'
                         '    printf("*** PASS ***\\n\\n");\n'
                         '  else\n'
                         '    printf("!!! FAIL !!!\\n\\n");\n\n')

            c_file.write('  printf("Output of final layer:\\n");\n'
                         '  for (i = 0; i < output_size; i++) {\n')
            if final_size == 7:
                c_file.write('    printf("%5hhd", (int8_t) (output[i] & 0xff));\n')
            else:
                c_file.write('    printf("%8d", (int32_t) output[i]);\n')
            c_file.write('    if ((i + 1) % 32 == 0)\n      printf("\\n");\n'
                         '    else if ((i + 1) % 4 == 0)\n      printf(" ");\n'
                         '  }\n'
                         '  printf("\\n");\n'
                         '\n')

            c_file.write('  return 0;\n}\n\n')

        # Close header files
        sampledata_header.close()
        weight_header.close()

        assets.copy('assets', 'cmsis-nn', base_directory, test_name)

        print(stats.summary())

        return test_name
