###################################################################################################
# Copyright (C) Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################
"""
ONNX File Routines
"""
import sys
import torch
import numpy as np
import onnx
import onnx.shape_inference
from onnx import numpy_helper

import op as opn
import tornadocnn
from eprint import eprint

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

def get_datatype(data):
    """
    internal print data type
    """
    print("data type:")
    data_type_s = [
        "unknown",
        "float32",
        "uint8",
        "int8",
        "uint16",
        "int16",
        "int32",
        "int64",
        "string",
        "bool",
        "float16",
        "double",
        "uint32",
        "uint64",
        "complex64",
        "complex128"
    ]
    print(data_type_s[data.data_type])

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

def eight_bit_quantitize(w_in,scale_factor):
    """quantitize to 8 bit as scale and zeropoint"""
    low = np.min(w_in)
    high = np.max(w_in)
    #scale = (high - low) / 256.
    scale = (high - low) / (128.0*scale_factor) ##
    #w = (w_in - low) / scale
    w = (w_in) / scale ##
    w_out = w.astype("int8")##
    return w_out, low, scale

def basic_quantize0(w_in,first,scale_factor):
    w,l,s = eight_bit_quantitize(w_in,scale_factor)
    return w

def basic_quantize(w,first,scale_factor):
    """
    checkpoint quantize algorithm
    """
    wscale=128*scale_factor
    if first:
        wscale /= 2

    w=w*wscale
    w=np.clip(w,-128,127).round() # FIXME - base on quantize bits
    w=w.astype(np.int64)
    
    return w

def get_perm(index,perm):
    """
    """
    count = 0
    for _index in perm:
        if _index == index:
            return count
        count = count + 1;

    return -1

def process_channels(model,_input,initializers,do_quant,div,scale_factor):
    """
    Match model and initializer names from input to find weights.
    """
    internal_quantized = False
    w = None

    if _input in initializers:
        for _init in model.graph.initializer:
            if _input == _init.name:
                if len(numpy_helper.to_array(_init).shape) == 1:
                    # bias
                    w = numpy_helper.to_array(_init)
                elif isinstance(_init, np.int64):
                    w = numpy_helper.to_array(_init).astype(np.int64)
                elif do_quant == True:
                    w = basic_quantize(numpy_helper.to_array(_init),div,scale_factor)
                    internal_quantized = True
                else:
                    w = numpy_helper.to_array(_init).astype(np.int64)
                break
    return w, internal_quantized

def manage_bias(
        w, bias, bias_min, bias_max, bias_keys, bias_quant, bias_size, param_size, bias_quantization, seq, _input, param_count, do_quant, first_bias,scale_factor
):
    '''
    Collect bias info. Verify range. Quantize if needed. Modularized repeatitive code.
    '''
    if do_quant == False:  # using fixed bias quant
        #w = w // tornadocnn.dev.BIAS_DIV
        w = w 
    else:
        ''' 
        if weights are not quantized we do not want to divide weights

        from quantize.py:
          # Save conv biases so PyTorch can still use them to run a model. This needs
          # to be reversed before loading the weights into the AI84/AI85.
          # When multiplying data with weights, 1.0 * 1.0 corresponds to 128 * 128 and
          # we divide the output by 128 to compensate. The bias therefore needs to be
          # multiplied by 128. This depends on the data width, not the weight width,
          # and is therefore always 128.
          if dev != 84:
              weights *= 2**(tc.dev.ACTIVATION_BITS-1)

          from checkpoint.py:
          if bias_name in checkpoint_state and seq not in no_bias:
              w = checkpoint_state[bias_name].numpy(). \
              astype(np.int64) // tornadocnn.dev.BIAS_DIV

        ''' 
        w = basic_quantize(w,False,scale_factor)

    w=w.astype(np.int64)

    w_min, w_max = w.min(), w.max()
    assert w_min >= -(2**(bias_quantization[seq]-1)),print(w_min)
    assert w_max < 2**(bias_quantization[seq]-1),print(w_max)

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

def load(
        checkpoint_file,
        unused_arch,
        fc_layer,
        quantization,
        bias_quantization,
        kernel_size,  # this information available in onnx model
        operator,
        verbose=False,
        no_bias=None,
        scale=None,
):
    """
    Load weights and biases from `checkpoint_file`. If `arch` is not None and does not match
    the architecuture in the checkpoint file, abort with an error message. If `fc_layer` is
    `True`, configure a single fully connected classification layer for software rather than
    hardware.
    `quantization` is a list of expected bit widths for the layer weights (always 8 for AI84).
    This value is checked against the weight inputs.
    `bias_quantization` is a list of the expected bit widths for the layer weights (always
    8 for AI84/AI85).
    In addition to returning weights and biases, this function configures the network output
    channels and the number of layers.
    When `verbose` is set, display the shapes of the weights.
    """

    # Set to False to make results of pytorch onnx export match quantized checkpoint results
    # Set to True for unquantized TF onnx exports
    do_quantize = False
    scale_factor = 1.0

    if scale is not None:
        do_quantize = True
        scale_factor = float(scale)

    first = True
    first_bias = True

    model = onnx.load(checkpoint_file)
    print(f'Reading {checkpoint_file} to configure network weights...')

    #print(model)
    #print(onnx.helper.printable_graph(model.graph,prefix=''))

    layers = 0
    num_conv_layers = len(quantization)
    no_bias = no_bias or []
    weights = []
    temp_weight = []
    w_last = []
    bias = []
    fc_weights = []
    fc_bias = []
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
    is_not_layer = 0
    kernel_size_onnx = []
    input_dims = []
    output_dims = []
    _dim = []
    matmul_out = None
    transposed = False
    trans_perm = (0,1,2,3)
    trans_bias = False
    oplist = []
    t0 = 0
    t1 = 1
    t2 = 2
    t3 = 3
    squeeze = False
    unsqueeze = False
    unsqz_dim = -1
    sqz_dim = -1
    constant = np.empty(2)
    shape_val = None
    num_layer_ops = 0
    last_output = ''
    #print(model.graph.initializer)
    initializers = {t.name for t in model.graph.initializer}
    cast_out = None
    mul_out = None
    add_out = None
    cast_w = []
    cast_ref = []
    cast_w_quant = []
    cast_out_ref = []
    cast_ref_index = 0
    mul_in = None
    add_in = None

    # find Cast/Mul/Add sequences with connected in/outs 
    # and integer initializers in Cast node
    # according to quantize script, Cast initializer is quantized weights
    # and Add/Mul initializers are dequantize operands
    for _, node in enumerate(model.graph.node):
        if node.op_type == 'Cast' or node.op_type == 'Mul' or node.op_type == 'Add':
            _inputs, _outputs = get_inouts(node)
            if node.op_type == 'Cast':
                cast_out = _outputs[0]
                for attr in node.attribute:
                    if attr.name == 'to':
                        if attr.HasField("i"):
                           if attr.i == 1:
                               cast_w, iq = process_channels(model,_inputs[0],initializers, False, False,scale_factor)
                               #print("CAST_W")
                               #print(cast_w)
                               cast_w = cast_w.astype(np.int8)
                               cast_w=np.clip(cast_w,-128,127).round()

            if node.op_type == 'Mul':
                mul_out = _outputs[0]
                mul_in = _inputs[0]
                
            if node.op_type == 'Add':
                add_out = _outputs[0]
                add_in = _inputs[0]

                if (cast_out == mul_in) and (mul_out == add_in):
                    cast_w_quant.append(cast_w)
                    cast_out_ref.append(add_out)
                    cast_out = None
                    mul_out = None
                    add_out = None
                    cast_w = []
                    mul_in = None
                    add_in = None
                    #print(cast_w_quant)
        
    for _, node in enumerate(model.graph.node):
        oplist.append(node.op_type)

        _inputs, _outputs = get_inouts(node)

        if node.op_type == 'Cast':
            continue

        if node.op_type == 'Mul':
            continue

        if node.op_type == 'Conv' or node.op_type == 'Gemm' \
           or node.op_type == 'MatMul' or  node.op_type == 'Add':
            #print("IN_OUT")
            #print(_inputs)
            #print(_outputs)
            if node.op_type == 'Conv' or node.op_type == 'Gemm' \
               or node.op_type == 'MatMul':
                num_layer_ops += 1
                if node.op_type == 'MatMul':
                    matmul_out = _outputs[0]  # reference to find following Add(matmul_out,bias) 
                    #print("MMO")
                    #print(matmul_out)
 
                for _input in _inputs:
                    if _input in initializers:
                        for _init in model.graph.initializer:
                            if _input == _init.name:
                                for _dim in _init.dims:
                                    input_dims.append(_dim)

            if node.op_type == 'Gemm' or  node.op_type == 'MatMul':
                kernel_shape = [1, 1]
                kernel_size_onnx.append(kernel_shape)

                if layers >= num_conv_layers:
                    continue

            if node.op_type == 'Conv':
                for a in node.attribute:
                    if a.name == 'kernel_shape':
                        if len(a.ints) == 1:
                            kernel_shape = [a.ints[0], 1]
                            kernel_size_onnx.append(kernel_shape)
                        else:
                            kernel_size_onnx.append(a.ints)

            for _input in _inputs:
                #cast_w = cast_w_quant[cast_ref_index]
                #cast_ref = cast_out_ref[cast_ref_index]
                w,internal_quantized=process_channels(model,_input,initializers,do_quantize,first,scale_factor)
                #eprint("WWWWWWWWW")
                #eprint(node.op_type)
                #eprint(w)
                if internal_quantized == True:
                    if len(w.shape) > 1:
                        first = False

                if w is None:
                    # tf quantize script stores weights in Cast node
                    if _input in cast_out_ref:
                        index = 0
                        for cast_ref in cast_out_ref:
                            #print(cast_ref)
                            if _input == cast_ref:
                                w = cast_w_quant[index]
                                #print("USING_Q_WEIGHT")
                                break
                            index = index + 1

                if w is not None:
                    if node.op_type == 'Gemm' or  node.op_type == 'MatMul' \
                       or node.op_type == 'Add':  # general matrix multiplication (FC layer)
                        if node.op_type == 'Gemm' or  node.op_type == 'MatMul':
                            temp_weight = w
                            if fc_layer:
                                if _input == _inputs[1]:  # weight
                                    assert w.min() >= -128 and w.max() <= 127
                                    #eprint("FC WEIGHTS")
                                    fc_weights.append(w)

                                if node.op_type == 'Gemm':
                                    if len(_inputs) == 3:  # have optional bias input
                                        if _input == _inputs[2]:  # bias
                                            assert w.min() >= -128 and w.max() <= 127
                                            fc_bias.append(w)
                                    elif _input == _inputs[1]:  # add bias 'None'
                                        fc_bias.append(None)    # during weight input processing

                        if node.op_type == 'Add':
                            #eprint("ADD:")

                            if _inputs[0] == matmul_out:
                                #eprint("MATMUL_OUT")
                                if fc_layer:
                                    if _input == _inputs[1]:  # bias
                                        #eprint("ADD FC BIAS")
                                        assert w.min() >= -128 and w.max() <= 127
                                        fc_bias.append(w)
                                else:
                                    #eprint("ADD NOT FC")

                                    if len(bias) == seq:
                                        del bias[seq-1] # remove default matmul bias entries if bias/Add detected
                                        del bias_min[seq-1]
                                        del bias_max[seq-1]
                                        del bias_keys[seq-1]
                                        del bias_quant[seq-1]
                                        del bias_size[seq-1]
                                    
                                    manage_bias(w,bias,bias_min,bias_max,bias_keys,bias_quant,bias_size,param_size,bias_quantization,seq-1,_input,param_count,do_quantize, False,scale_factor)
                                    #eprint(bias)
                            first_bias = False
                            is_not_layer = 1
                            continue

                    if len(w.shape) > 1:  # not a bias
                        quant.append(quantization[seq])

                        w_min, w_max = w.min(), w.max()
                        assert w_min >= -(2**(quantization[seq]-1)), print(w_min)
                        assert w_max < 2**(quantization[seq]-1), print(w_max)
                        w=w.astype(np.int64)

                        weight_min.append(w_min)
                        weight_max.append(w_max)

                        # TODO: Double check if we need to check conv2d if opn is known
                        # to be opn.CONVTRANSPOSE2D. We should be able to get this
                        # from the op_type Conv plus shape?
                        if operator[seq] == opn.CONVTRANSPOSE2D:
                            # For ConvTranspose2d, flip the weights as follows:
                            w = np.flip(w, axis=(2, 3)).swapaxes(0, 1)

                        if len(w.shape) == 2:
                            if w.shape[t0] >  w.shape[t1]:
                                trans_perm = (t1,t0)
                                w = np.transpose(w, (trans_perm))

                        input_channels.append(w.shape[t1])  # Input channels
                        output_channels.append(w.shape[t0])  # Output channels

                        if len(w.shape) == 2:  # MLP
                            if kernel_size_onnx[seq][0] != 1 or kernel_size_onnx[seq][1] != 1:
                                eprint(f'The `kernel_size` for the MLP layer {seq} should '
                                       f'be set to 1x1 instead of '
                                       f'{kernel_size_onnx[seq][0]}x{kernel_size_onnx[seq][1]}.')
                                error_exit = True
                        elif len(w.shape) == 3:  # 1D
                            #eprint(w.shape)
                            #eprint(kernel_size_onnx[seq])
                            #eprint(kernel_size)
                            if kernel_size_onnx[seq][0] != w.shape[2] \
                               or kernel_size_onnx[seq][1] != 1:
                                eprint(f'The `kernel_size` for the 1D layer {seq} should '
                                       f'be set to {w.shape[2]}x1 instead of '
                                       f'{kernel_size_onnx[seq][0]}x{kernel_size_onnx[seq][1]}.')
                                error_exit = True
                        elif len(w.shape) == 4:  # 2D
                            if kernel_size_onnx[seq][0] != w.shape[t2] \
                               or kernel_size_onnx[seq][1] != w.shape[t3]:
                                #eprint(w.shape)
                                eprint(f'The `kernel_size` for the 2D layer {seq} should '
                                       f'be set to {w.shape[2]}x{w.shape[3]} instead of '
                                       f'{kernel_size[seq][0]}x{kernel_size[seq][1]}.')
                                error_exit = True

                        w_count = np.prod(w.shape)
                        param_count += w_count
                        w_size = (w_count * 8 + (quantization[seq]-1)) // quantization[seq]
                        weight_size.append(w_size)
                        param_size += w_size

                        w_last = w

                        if len(w.shape) == 2:  # linear - add dummy 'channel'
                            #print("Dummy")
                            w = np.expand_dims(w, axis=0)
                        else:  # conv1d, conv2d, ... - combine input and output channels
                            #print("Slice and dice")
                            if squeeze == True:
                                w = np.reshape(w, (-1, ) + w.shape[sqz_dim:])
                            else:
                                w = np.reshape(w, (-1, ) + w.shape[2:])
                        #eprint("WAPPEND")
                        #eprint(seq)
                        weights.append(w)
                        weight_keys.append(_input)
                        #print("SHAPE")
                        #print(w.shape)
                    #print("BIAS?")
                    #print(_input)
                    #print(cast_ref)
                    #if len(_inputs) > 2:
                        #print(_inputs[2])

                    #print("bias?")
                    if len(_inputs) < 3 or \
                       ((_input == _inputs[2] or cast_ref == _inputs[2]) and seq in no_bias):  # no bias input
                        #eprint("XXXXXXXXXXXXXXXXXX")
                        #print(len(_inputs))
                        if len(_inputs) > 32:
                            print(_inputs[2])
                        #print(seq)
                        #print(no_bias)
                        bias.append(None)
                        bias_min.append(0)
                        bias_max.append(0)
                        bias_keys.append('N/A')
                        bias_quant.append(0)
                        bias_size.append(0)
                    elif _input == _inputs[2]:  # bias input
                        #eprint("xxxxXXXXXXXXXXXXXX")
                        #eprint(seq)
                        #eprint(w)
                        manage_bias(w,bias,bias_min,bias_max,bias_keys,bias_quant,bias_size,param_size,bias_quantization,seq,_input,param_count,do_quantize,False,scale_factor) #first_bias) #internal_quantized)
                        #eprint(bias)
                        first_bias = False

            if is_not_layer == 0:
                seq += 1
                layers += 1

            transposed = False
            trans_perm = (0,1,2,3)
            t0 = get_perm(0,trans_perm)
            t1 = get_perm(1,trans_perm)
            t2 = get_perm(2,trans_perm)
            t3 = get_perm(3,trans_perm)
            squeeze = False
            unsqueeze = False

            is_not_layer = 0

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
        print(f'TOTAL: {layers} layers, {param_count:,} parameters, {param_size:,} bytes')

    if error_exit:
        sys.exit(1)

    if verbose:
    #if not verbose:
        with np.printoptions(threshold=np.inf, linewidth=80):
            print("\nSUMMARY\n=======")
            print(layers, "layers\n")
            print("weights:")
            print(weights)
            print("bias:")
            print(bias)
            print("fc_weights:")
            print(fc_weights)
            print("fc_bias:")
            print(fc_bias)
            print("input_channels:")
            print(input_channels)
            print("output_channels:")
            print(output_channels)
            print("len(weights):")
            print(len(weights))
            print("")
            print(_dim)
            print("")
            print(oplist)

    return layers, weights, bias, fc_weights, fc_bias, input_channels, output_channels
