###################################################################################################
# Copyright (C) 2019 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Confidential
#
# Written by RM
###################################################################################################
"""
Routines to generate software CNNs using Arm's CMSIS NN library
"""
import os
import sys
import numpy as np
import toplevel
from simulate import cnn1d_layer, cnn2d_layer, linear_layer


def create_net(prefix, verbose, debug, log,
               layers, convolution, input_dim, pooled_dim, output_dim,
               input_size, kernel_size,
               quantization, input_chan, output_chan, output_width, padding, dilation, stride,
               pool, pool_stride, pool_average, activate,
               data, kernel, bias, fc_weights, fc_bias,
               c_filename, base_directory, log_filename,
               weight_filename, sample_filename, ai85=False):
    """
    Create the CMSIS NN network.
    """
    if any(w != 8 for w in output_width):
        print('CMSIS network generator does not currently support `output_width` that is not 8.')
        sys.exit(1)
    if any(c != 2 for c in convolution):
        print('CMSIS network generator does not currently support `convolution` that is not 2D.')
        sys.exit(1)

    test_name = prefix
    print(f'{test_name}...')

    os.makedirs(os.path.join(base_directory, test_name), exist_ok=True)

    # Redirect stdout?
    if log:
        sys.stdout = open(os.path.join(base_directory, test_name, log_filename), 'w')
        print(f'{test_name}')

    filename = c_filename + '.c'
    sampledata_header = \
        open(os.path.join(base_directory, test_name, sample_filename), mode='w')
    weight_header = \
        open(os.path.join(base_directory, test_name, weight_filename), mode='w')

    with open(os.path.join(base_directory, test_name, filename), mode='w') as c_file:
        toplevel.copyright_header(c_file)

        c_file.write(f'// {test_name}\n')
        c_file.write(f'// Created using {" ".join(str(x) for x in sys.argv)}\n')

        # Human readable description of test
        c_file.write(f'\n// Configuring {layers} layer{"s" if layers > 1 else ""}:\n')

        for ll in range(layers):
            c_file.write(f'// Layer {ll+1}: '
                         f'{input_chan[ll]}x{input_dim[ll][0]}x{input_dim[ll][1]}, ')
            if pool[ll] > 0:
                c_file.write(f'{pool[ll]}x{pool[ll]} {"avg" if pool_average[ll] else "max"} '
                             f'pool with stride {pool_stride[ll]}')
            else:
                c_file.write(f'no pooling')
            c_file.write(f', {kernel_size[ll][0]}x{kernel_size[ll][1]} convolution '
                         f'with stride {stride[ll]} '
                         f'pad {padding[ll]}, '
                         f'{output_chan[ll]}x{output_dim[ll][0]}x{output_dim[ll][1]} out\n')

        c_file.write('\n')
        toplevel.header(c_file, 0, embedded_code=True, cmsis_nn=True)

        # Pre-define data memory loader.
        d = data.transpose((1, 2, 0)).flatten()
        toplevel.c_define(sampledata_header, d, 'INPUT_DATA', '%d', 16)
        c_file.write(f'static const q7_t input_data[{d.size}] = INPUT_DATA;\n')

        # Pre-define the kernels and bias values
        for ll in range(layers):
            w = kernel[ll]. \
                reshape((output_chan[ll], input_chan[ll],
                         kernel_size[ll][0], kernel_size[ll][1])). \
                transpose((0, 2, 3, 1)). \
                flatten()
            toplevel.c_define(weight_header, w, f'WEIGHTS_{ll}', '%d', 16)
            if bias[ll]:
                b = bias[ll].flatten()
            else:
                # We need empty bias values (the Arm code needs them both for rounding of
                # the shifted output, and it does not like NULL bias pointers)
                b = np.zeros(output_chan[ll], dtype=np.int64)
            toplevel.c_define(weight_header, b, f'BIAS_{ll}', '%d', 16)
        c_file.write('\n')

        for ll in range(layers):
            c_file.write(f'static const q7_t weights_{ll}[{kernel[ll].size}] = '
                         f'WEIGHTS_{ll};\n')
            c_file.write(f'static const q7_t bias_{ll}[{output_chan[ll]}] = '
                         f'BIAS_{ll};\n')
        c_file.write('\n')

        # Compute buffer sizes
        col_buffer_size = 0
        img_buffer_size = 0
        for ll in range(layers):
            col_buffer_size = max(col_buffer_size,
                                  2*input_chan[ll]*kernel_size[ll][0]*kernel_size[ll][1])
            img_buffer_size = max(img_buffer_size,
                                  input_chan[ll]*input_dim[ll][0]*input_dim[ll][1])

        c_file.write(f'static q7_t buffer0[{img_buffer_size}];\n')
        c_file.write(f'static q7_t buffer1[{img_buffer_size}];\n')
        c_file.write(f'static q15_t col_buffer[{col_buffer_size}];\n\n')

        c_file.write('int cnn_run(void)\n{\n')

        # Compute layer-by-layer output and chain results into input
        buffer0 = 'buffer0'
        buffer1 = 'buffer1'

        for ll in range(layers):
            c_file.write(f'  // Layer {ll}: {input_size} -> ')
            if convolution[ll] == 2:
                if pool[ll]:
                    c_file.write(f'[{input_size[0]}, {pooled_dim[ll][0]}, '
                                 f'{pooled_dim[ll][1]}] -> ')
                out_buf, out_size = cnn2d_layer(ll + 1, verbose,
                                                input_size, kernel_size[ll], quantization[ll],
                                                output_chan[ll],
                                                [padding[ll], padding[ll]], dilation[ll],
                                                [stride[ll], stride[ll]],
                                                [pool[ll], pool[ll]],
                                                [pool_stride[ll], pool_stride[ll]],
                                                pool_average[ll],
                                                activate[ll],
                                                kernel[ll].reshape(output_chan[ll], input_size[0],
                                                                   kernel_size[ll][0],
                                                                   kernel_size[ll][1]),
                                                bias[ll],
                                                data,
                                                ai85=ai85,
                                                debug=debug)
            else:
                if pool[ll]:
                    c_file.write(f'[{input_size[0]}, {pooled_dim[ll][0]}] -> ')
                out_buf, out_size = cnn1d_layer(ll + 1, verbose,
                                                input_size, kernel_size[ll][0], quantization[ll],
                                                output_chan[ll],
                                                padding[ll], dilation[ll][0],
                                                stride[ll],
                                                pool[ll],
                                                pool_stride[ll],
                                                pool_average[ll][0],
                                                activate[ll],
                                                kernel[ll].reshape(output_chan[ll], input_size[0],
                                                                   kernel_size[ll][0]),
                                                bias[ll],
                                                data,
                                                ai85=ai85,
                                                debug=debug)
            c_file.write(f'{out_size}\n')

            if pool[ll]:
                if pool_average[ll]:
                    c_file.write(f'  arm_avepool_q7_HWC({buffer0}, {input_dim[ll][0]}, '
                                 f'{input_chan[ll]}, {pool[ll]}, 0, {pool_stride[ll]}, '
                                 f'{pooled_dim[ll][0]}, NULL, {buffer1});\n')
                else:
                    c_file.write(f'  arm_maxpool_q7_HWC({buffer0}, {input_dim[ll][0]}, '
                                 f'{input_chan[ll]}, {pool[ll]}, 0, {pool_stride[ll]}, '
                                 f'{pooled_dim[ll][0]}, NULL, {buffer1});\n')
                n = buffer0
                buffer0 = buffer1
                buffer1 = n

            source = 'input_data' if ll == 0 else buffer0
            fn = 'fast' if input_chan[ll] % 4 == 0 and output_chan[ll] % 2 == 0 else 'basic'
            c_file.write(f'  arm_convolve_HWC_q7_{fn}({source}, {pooled_dim[ll][0]}, '
                         f'{input_chan[ll]}, weights_{ll}, {output_chan[ll]}, '
                         f'{kernel_size[ll][0]}, '
                         f'{padding[ll]}, {stride[ll]}, bias_{ll}, 0, 7, {buffer1}, '
                         f'{output_dim[ll][0]}, col_buffer, NULL);\n')

            if activate[ll]:
                c_file.write(f'  arm_relu_q7({buffer1}, '
                             f'{output_dim[ll][0] * output_dim[ll][1] * output_chan[ll]});\n')
            n = buffer0
            buffer0 = buffer1
            buffer1 = n

            input_size = [out_size[0], out_size[1], out_size[2]]
            data = out_buf.reshape(input_size[0], input_size[1], input_size[2])
            c_file.write('\n')
            data_cmsis = data.transpose((1, 2, 0)).flatten()
            if verbose:
                print('TRANSPOSED AND FLATTENED:')
                print(data_cmsis)
                print('')

        c_file.write('  return 1;\n}\n\n')

        if fc_weights:
            data = data.flatten()

            out_buf, out_size = linear_layer(verbose=verbose,
                                             do_activation=False,
                                             data=data, weight=fc_weights[0], bias=fc_bias[0],
                                             debug=debug)

            # Rearrange the weights to account for the shape of the conv layer output
            w = fc_weights[0]. \
                reshape((fc_weights[0].shape[0], input_size[0], input_size[1], input_size[2])). \
                transpose(0, 2, 3, 1). \
                reshape((fc_weights[0].shape[0], fc_weights[0].shape[1]))

            # np.dot(worg, torg.flatten()) should be equal to np.dot(wnew, tnew.flatten())
            assert (np.dot(fc_weights[0], data) == np.dot(w, data_cmsis)).all()

            toplevel.fc_layer(c_file, weight_header, w, fc_bias[0], cmsis_nn=True)

        c_file.write('int main(void)\n{\n')
        if fc_weights:
            c_file.write('  int i;\n\n')
        c_file.write('  cnn_run();\n')
        if fc_weights:
            c_file.write(f'  fc_layer({buffer0});\n\n')
            c_file.write('  printf("Classification results:\\n");\n'
                         '  for (i = 0; i < FC_OUT; i++) {\n'
                         '    printf("[%6d] -> Class %d: %0.1f%%\\n", fc_output[i], i, '
                         '(double) (100.0 * fc_softmax[i] / 32768.0));\n'
                         '  }\n\n')
        c_file.write('  return 0;\n}\n\n')

    # Close header files
    sampledata_header.close()
    weight_header.close()
