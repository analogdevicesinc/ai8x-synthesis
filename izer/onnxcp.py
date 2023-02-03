###################################################################################################
# Copyright (C) 2020-2021 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################
"""
ONNX File Routines
"""
import sys

import numpy as np

import onnx
import onnx.shape_inference
from onnx import numpy_helper

from . import op as opn
from . import tornadocnn as tc
from .eprint import eprint
from .utils import fls


def get_attribute(attr):
    """
    Return name and data from attribute.
    """
    data = None
    if attr.HasField("f"):
        data = attr.f
    elif attr.HasField("i"):
        data = attr.i
    elif attr.HasField("s"):
        data = attr.s
    elif attr.HasField("t"):
        data = attr.t
    elif attr.HasField("g"):
        data = attr.g
    elif attr.floats:
        data = attr.floats
    elif attr.ints:
        data = attr.ints
    elif attr.strings:
        data = attr.strings
    elif attr.tensors:
        data = attr.tensors
    elif attr.graphs:
        data = attr.graphs
    return attr.name, data


def get_inouts(node):
    """
    Return list of inputs and outputs.
    """
    inputs = []
    for i in node.input:
        inputs.append(i)

    outputs = []
    for i in node.output:
        outputs.append(i)

    return inputs, outputs


def process_channels(model, _input, initializers):
    """
    Match model and initializer names from input to find weights.
    """
    if _input in initializers:
        for _init in model.graph.initializer:
            if _input == _init.name:
                w = numpy_helper.to_array(_init).astype(np.int64)
                break
    else:
        w = None
    return w


def load(
        checkpoint_file,
        unused_arch,
        quantization,
        bias_quantization,
        output_shift,
        kernel_size,  # this information available in onnx model
        operator,
        verbose=False,
        no_bias=None,
):
    """
    Load weights and biases from `checkpoint_file`. If `arch` is not None and does not match
    the architecture in the checkpoint file, abort with an error message.
    `quantization` is a list of expected bit widths for the layer weights.
    This value is checked against the weight inputs.
    `bias_quantization` is a list of expected bit widths for the layer bias weights (always 8).
    In addition to returning weights and biases, this function configures the network output
    channels and the number of layers.
    When `verbose` is set, display the shapes of the weights.
    """
    model = onnx.load(checkpoint_file)
    print(f'Reading {checkpoint_file} to configure network weights...')

    layers = 0
    num_conv_layers = len(quantization)
    no_bias = no_bias or []
    weights = []
    bias = []
    weight_keys = []
    bias_keys = []
    output_channels = []
    input_channels = []
    param_count = 0
    param_size = 0
    error_exit = False
    quant = []
    bias_quant = []
    weight_min = []
    weight_max = []
    weight_size = []
    bias_min = []
    bias_max = []
    bias_size = []
    seq = 0

    kernel_size_onnx = []

    initializers = {t.name for t in model.graph.initializer}
    for node in model.graph.node:

        if node.op_type in ('Conv', 'Gemm'):
            _inputs, _outputs = get_inouts(node)
            for _input in _inputs:
                w = process_channels(model, _input, initializers)
                if w is not None:
                    if node.op_type == 'Gemm':  # general matrix multiplication (FC layer)
                        kernel_shape = [1, 1]
                        kernel_size_onnx.append(kernel_shape)
                        if layers >= num_conv_layers:
                            continue

                    if node.op_type == 'Conv':  # (Conv layer)
                        for a in node.attribute:
                            if a.name == 'kernel_shape':
                                kernel_size_onnx.append(a.ints)

                    if w.ndim > 1:  # not a bias
                        quant.append(quantization[seq])

                        w_min, w_max = w.min(), w.max()

                        # Determine quantization or make sure that what was given fits
                        if quantization[seq] is not None:
                            assert w_min >= -(2**(quantization[seq]-1)), print(w_min)
                            assert w_max < 2**(quantization[seq]-1), print(w_max)
                        else:
                            if w_max > 0:
                                w_max_m = int(w_max)
                            else:
                                w_max_m = int(abs(w_max)) - 1
                            if w_min > 0:
                                w_min_m = int(w_min)
                            else:
                                w_min_m = int(abs(w_min)) - 1
                            quantization[seq] = 1 << (fls(max(fls(w_max_m), fls(w_min_m)) + 1) + 1)
                            assert quantization[seq] <= 8

                        weight_min.append(w_min)
                        weight_max.append(w_max)

                        # Not overriding output_shift?
                        if output_shift[seq] is None:
                            output_shift[seq] = 0
                        # Add based on quantization
                        output_shift[seq] += 8 - quantization[seq]

                        # TODO: Double check if we need to check conv2d if opn is known
                        # to be opn.CONVTRANSPOSE2D. We should be able to get this
                        # from the op_type Conv plus shape?
                        if operator[seq] == opn.CONVTRANSPOSE2D:
                            # For ConvTranspose2d, flip the weights as follows:
                            w = np.flip(w, axis=(2, 3)).swapaxes(0, 1)

                        input_channels.append(w.shape[1])  # Input channels
                        output_channels.append(w.shape[0])  # Output channels

                        if w.ndim == 2:  # MLP
                            if kernel_size_onnx[seq][0] != 1 or kernel_size_onnx[seq][1] != 1:
                                eprint(f'The `kernel_size` for the MLP layer {seq} should '
                                       f'be set to 1x1 instead of '
                                       f'{kernel_size[seq][0]}x{kernel_size[seq][1]}.',
                                       exit_code=None)
                                error_exit = True
                        elif w.ndim == 3:  # 1D
                            if kernel_size_onnx[seq][0] != w.shape[2] \
                               or kernel_size_onnx[seq][1] != 1:
                                eprint(f'The `kernel_size` for the 1D layer {seq} should '
                                       f'be set to {w.shape[2]}x1 instead of '
                                       f'{kernel_size[seq][0]}x{kernel_size[seq][1]}.',
                                       exit_code=None)
                                error_exit = True
                        elif w.ndim == 4:  # 2D
                            if kernel_size_onnx[seq][0] != w.shape[2] \
                               or kernel_size_onnx[seq][1] != w.shape[3]:
                                eprint(f'The `kernel_size` for the 2D layer {seq} should '
                                       f'be set to {w.shape[2]}x{w.shape[3]} instead of '
                                       f'{kernel_size[seq][0]}x{kernel_size[seq][1]}.',
                                       exit_code=None)
                                error_exit = True

                        w_count = np.prod(w.shape)
                        param_count += w_count
                        w_size = (w_count * quantization[seq] + 7) // 8
                        weight_size.append(w_size)
                        param_size += w_size

                        if w.ndim == 2:  # linear - add dummy 'channel'
                            w = np.expand_dims(w, axis=0)
                        else:  # conv1d, conv2d, ... - combine input and output channels
                            w = np.reshape(w, (-1, ) + w.shape[2:])

                        weights.append(w)
                        weight_keys.append(_input)

                    if len(_inputs) < 3 or \
                       (_input == _inputs[2] and seq in no_bias):  # no bias input
                        bias.append(None)
                        bias_min.append(0)
                        bias_max.append(0)
                        bias_keys.append('N/A')
                        bias_quant.append(0)
                        bias_size.append(0)
                    elif _input == _inputs[2]:  # bias input
                        w = w // tc.dev.BIAS_DIV
                        w_min, w_max = w.min(), w.max()
                        assert w_min >= -(2**(bias_quantization[seq]-1))
                        assert w_max < 2**(bias_quantization[seq]-1)
                        bias_min.append(w_min)
                        bias_max.append(w_max)

                        bias.append(w)
                        bias_keys.append(_input)
                        bias_quant.append(bias_quantization[seq])
                        w_count = np.prod(w.shape)
                        param_count += w_count
                        w_size = (
                            w_count * 8 + (bias_quantization[seq]-1)
                        ) // bias_quantization[seq]
                        bias_size.append(w_size)
                        param_size += w_size

            seq += 1
            layers += 1
        # TODO: Things to add
        # if attribute.name == 'pads':
        # if attribute.name == 'strides':

    if verbose:
        print('Layer  InCh OutCh  Weights         Quant  Min Max   Size '
              'Key                                 Bias       Quant  Min Max Size Key')
        for ll in range(layers):
            if ll < len(weights) and weights[ll] is not None:
                weight_shape = str(weights[ll].shape)
                if bias[ll] is not None:
                    bias_shape = str(bias[ll].shape)
                else:
                    bias_shape = 'N/A'
                print(f'{ll:4}: '
                      f'{input_channels[ll]:5} {output_channels[ll]:5}  '
                      f'{weight_shape:15} '
                      f'{quant[ll]:5} {weight_min[ll]:4} {weight_max[ll]:3} {weight_size[ll]:6} '
                      f'{weight_keys[ll]:35} '
                      f'{bias_shape:10} '
                      f'{bias_quant[ll]:5} {bias_min[ll]:4} {bias_max[ll]:3} {bias_size[ll]:4} '
                      f'{bias_keys[ll]:25}')
        print(f'TOTAL: {layers} parameter layers, {param_count:,} parameters, '
              f'{param_size:,} bytes')

    if error_exit:
        sys.exit(1)

    if verbose:
        with np.printoptions(threshold=np.inf, linewidth=80):
            print("\nSUMMARY\n=======")
            print(layers, "layers\n")
            print("weights:")
            print(weights)
            print("bias:")
            print(bias)
            print("input_channels:")
            print(input_channels)
            print("output_channels:")
            print(output_channels)
            print("")

    return layers, weights, bias, output_shift, \
        input_channels, output_channels
