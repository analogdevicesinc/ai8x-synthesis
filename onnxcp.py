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

import numpy as np

import onnx
import onnx.shape_inference
import onnxruntime
from onnx import numpy_helper

import op as opn
from eprint import eprint
from utils import fls


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
        "complex128",
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


def eight_bit_quantitize(w_in, scale_factor):
    """
    quantitize to 8 bit as scale and zeropoint
    """
    low = np.min(w_in)
    high = np.max(w_in)
    scale = (high - low) / (128.0 * scale_factor)
    w = (w_in) / scale
    w_out = w.astype("int8")
    return w_out, low, scale


def basic_quantize0(
    w_in, first, scale_factor,  # pylint: disable=unused-argument
):
    """
    quantitize to 8 bit as scale and zeropoint
    """
    w, _, _ = eight_bit_quantitize(w_in, scale_factor)
    return w


def basic_quantize(w, first, scale_factor):
    """
    checkpoint quantize algorithm
    """
    wscale = 128 * scale_factor
    if first:
        wscale /= 2
    w = w * wscale
    w = np.clip(w, -128, 127).round()  # FIXME - base on quantize bits
    w = w.astype(np.int64)
    return w


def get_perm(index, perm):
    """
    get permutation count
    """
    count = 0
    for idx in perm:
        if idx == index:
            return count
        count += 1

    return -1


def process_channels(model, inp, initializers, do_quant, div, scale_factor):
    """
    Match model and initializer names from input to find weights.
    """
    internal_quantized = False
    w = None

    if inp in initializers:
        for initializer in model.graph.initializer:
            if inp == initializer.name:
                if len(numpy_helper.to_array(initializer).shape) == 1:
                    #  bias
                    w = numpy_helper.to_array(initializer)
                elif isinstance(initializer, np.int64):
                    w = numpy_helper.to_array(initializer).astype(np.int64)
                elif do_quant is True:
                    w = basic_quantize(
                        numpy_helper.to_array(initializer), div, scale_factor
                    )
                    internal_quantized = True
                else:
                    w = numpy_helper.to_array(initializer).astype(np.int64)
                break
    return w, internal_quantized


def manage_bias(
    w,
    bias,
    bias_min,
    bias_max,
    bias_keys,
    bias_quant,
    bias_size,
    param_size,
    bias_quantization,
    seq,
    inp,
    param_count,
    do_quant,
    first_bias,  # pylint: disable=unused-argument
    scale_factor,
):
    """
    Collect bias info. Verify range. Quantize if needed. Modularized repeatitive code.
    """
    if do_quant is False:  # using fixed bias quant
        pass
    else:
        # if weights are not quantized we do not want to divide weights
        # from quantize.py:
        # Save conv biases so PyTorch can still use them to run a model. This needs
        # to be reversed before loading the weights into the AI84/AI85.
        # When multiplying data with weights, 1.0 * 1.0 corresponds to 128 * 128 and
        # we divide the output by 128 to compensate. The bias therefore needs to be
        # multiplied by 128. This depends on the data width, not the weight width,
        # and is therefore always 128.
        # if dev != 84:
        #    weights *= 2**(tc.dev.ACTIVATION_BITS-1)
        # from checkpoint.py:
        # if bias_name in checkpoint_state and seq not in no_bias:
        #    w = checkpoint_state[bias_name].numpy(). \
        #    astype(np.int64) // tornadocnn.dev.BIAS_DIV
        w = basic_quantize(w, False, scale_factor)

    w = w.astype(np.int64)

    w_min, w_max = w.min(), w.max()
    assert w_min >= -(2 ** (bias_quantization[seq] - 1)), print(w_min)
    assert w_max < 2 ** (bias_quantization[seq] - 1), print(w_max)

    bias_min.append(w_min)
    bias_max.append(w_max)

    bias.append(w)
    bias_keys.append(inp)
    bias_quant.append(bias_quantization[seq])
    w_count = np.prod(w.shape)
    param_count += w_count
    w_size = (w_count * 8 + (bias_quantization[seq] - 1)) // bias_quantization[seq]
    bias_size.append(w_size)
    param_size += w_size


def track_data_shape(model, out_dict):
    """
    Trace data shape through the conv layers
    """
    layer_num = -1
    save_perm = []
    save_shape = []
    last_op = ""
    node_list = []
    conv_relu_pool = 0
    last_pool_crp = False

    for _, node in enumerate(model.graph.node):
        node_list.append(node.op_type)
        if node.op_type == "Conv":
            layer_num = layer_num + 1
            conv_relu_pool = 1
            if node.output[0] in out_dict.keys():
                save_shape = list(out_dict[node.output[0]].shape)

                if save_shape[0] == 1:  # set batch size to unknown
                    save_shape[0] = -1

        elif node.op_type == "ConvTranspose":
            layer_num = layer_num + 1
            if node.output[0] in out_dict.keys():
                save_shape = list(out_dict[node.output[0]].shape)
                if save_shape[0] == 1:  # set batch size to unknown
                    save_shape[0] = -1

        if node.op_type == "Relu":
            if conv_relu_pool == 1:
                conv_relu_pool = 2
            else:
                conv_relu_pool = 0

        elif node.op_type == "Squeeze":
            if last_op == "Conv" or (
                last_op in ("MaxPool", "AveragePool") and last_pool_crp
            ):
                for attr in node.attribute:
                    if attr.name == "axes":
                        if attr.ints:
                            save_shape.pop(attr.ints[0])

        elif (node.op_type == "MaxPool") or (node.op_type == "AveragePool"):
            last_pool_crp = False
            kernel = []
            stride = []
            for attr in node.attribute:
                if attr.name == "kernel_shape":
                    kernel_len = len(attr.ints)
                    for x in range(kernel_len):
                        kernel.append(attr.ints[x])
                if attr.name == "strides":
                    stride_len = len(attr.ints)
                    for x in range(stride_len):
                        stride.append(attr.ints[x])

            if conv_relu_pool == 2:
                last_pool_crp = True
                save_shape = list(out_dict[node.output[0]].shape)

                if save_shape[0] == 1:  # set batch size to unknown
                    save_shape[0] = -1

                conv_relu_pool = 0
                layer_num = layer_num + 1

        elif node.op_type == "Transpose":
            for attr in node.attribute:
                if attr.name == "perm":
                    perm_len = len(attr.ints)
                    perm = []
                    for x in range(perm_len):
                        perm.append(attr.ints[x])

            if len(perm) > 0:
                save_perm = perm.copy()

            if len(save_shape) > 0:
                temp = save_shape.copy()
                if len(save_perm) == len(save_shape):
                    for x in range(
                        len(save_perm)
                    ):  # pylint: disable=consider-using-enumerate
                        save_shape[x] = temp[save_perm[x]]

        elif node.op_type == "MatMul" or node.op_type == "Gemm":
            layer_num = layer_num + 1

        last_op = node.op_type
    return save_shape, save_perm


def inv(perm):
    """
    Determine the inverse permutation of the perm parameter
    """
    inverse = [0] * len(perm)
    for i, p in enumerate(perm):
        inverse[p] = i
    return inverse


def modify_weights(input_file, output_file, scale, first, dequantize):
    """
    pulls weights/biases out of initializers for Conv, Gemm and MatMul
    and quantizes them using scale factor provided
    """
    w = []
    last_op_type = None
    mat_mul_out = None
    model = onnx.load(input_file)
    onnx.checker.check_model(model)
    initializers = {t.name for t in model.graph.initializer}

    for _, node in enumerate(model.graph.node):
        if (
            node.op_type == "Conv"
            or node.op_type == "ConvTranspose"
            or node.op_type == "Gemm"
            or node.op_type == "MatMul"
            or node.op_type == "Add"
        ):
            inp = node.input[1]  # weights

            if node.op_type == "MatMul":
                mat_mul_out = node.output[0]

            if node.op_type == "Add":
                if last_op_type == "MatMul":
                    if node.input[0] == mat_mul_out:
                        mat_mul_out = None  # bias for MatMul
                    # allow other Add operations... they are probably biases

            if inp in initializers:
                init_count = 0
                for initializer in model.graph.initializer:
                    if inp == initializer.name:
                        w = numpy_helper.to_array(initializer)
                        w = basic_quantize(w, first, scale).astype(np.float32)
                        if dequantize is True:
                            w /= 128.0
                        model.graph.initializer[init_count].raw_data = w.tobytes()
                        first = False

                    if len(node.input) > 2:
                        input_b = node.input[2]  # bias for Conv, Gemm
                        if input_b == initializer.name:
                            w = numpy_helper.to_array(initializer)
                            w = basic_quantize(w, first, scale).astype(np.float32)
                            if dequantize is True:
                                w /= 128.0
                            model.graph.initializer[init_count].raw_data = w.tobytes()
                            first = False
                    init_count = init_count + 1

        last_op_type = node.op_type

    onnx.checker.check_model(model)
    onnx.save(model, output_file)


float_dict = {
    "tensor(float16)": "float16",
    "tensor(float)": "float32",
    "tensor(double)": "float64",
}

integer_dict = {
    "tensor(int32)": "int32",
    "tensor(int8)": "int8",
    "tensor(uint8)": "uint8",
    "tensor(int16)": "int16",
    "tensor(uint16)": "uint16",
    "tensor(int64)": "int64",
    "tensor(uint64)": "uint64",
}


def tensor_has_valid_type(tensor_valueproto, verbose):
    """
    Ensures ValueProto tensor element
    type is not UNDEFINED
    """  # FIXME pylint below
    if (
        tensor_valueproto.type.tensor_type.elem_type == onnx.TensorProto.UNDEFINED
    ):  # pylint: disable=no-member
        if verbose:
            print(
                "Type could not be inferred for the following output,"
                " it will be not be exposed:\n",
                tensor_valueproto,
            )
        return False
    return True


def expose_node_outputs(model_path, overwrite, run_checker=True, verbose=False):
    """
    Samrat Debroy,
    https://github.com/zetane/expose_onnx_node_outputs/blob/master/expose_node_outputs.py
    Exposes each intermediary node as one of the ONNX model's graph outputs.
     This allows inspection of data passing through the model.
    :param model_path: (str) The path to the .onnx model, with the extension.
    :param overwrite:  (boolean) If true, will overwrite the .onnx file at model_path,
                       else will make a copy.
    :param run_checker: (boolean) If true, will run the ONNX model validity checker
    :param verbose: (boolean) If true, will print detailed messages about execution
                              of preprocessing script
    :return: (str) file path to new ONNX model
    """
    # 1. Get name of all external outputs to the model (ie. graph-level outputs,
    #    not internal outputs shared bw nodes)
    model = onnx.load(model_path)
    external_outputs = [output.name for output in model.graph.output]
    extended_outputs = []

    # 2. Get the list of nodes in the graph
    for i, node in enumerate(model.graph.node):
        # 3. For every node, copy its (internal) output over to graph.output
        #    to make it a graph output
        output_name = [
            output for output in node.output if output not in external_outputs
        ]
        extended_outputs.extend(output_name)
        for output in output_name:
            # Added to expose Intermediate Node data
            intermediate_layer_value_info = onnx.helper.make_tensor_value_info(
                output,  # FIXME: linter below
                onnx.TensorProto.UNDEFINED,  # pylint: disable=E1101, # noqa:E501
                None,
                node.op_type,
            )
            model.graph.output.extend([intermediate_layer_value_info])

    if verbose:
        print(
            "The following nodes were exposed as outputs in the {} model:\n {}".format(
                model_path, extended_outputs
            )
        )

    # If all outputs were already "external", no changes are required to the ONNX model,
    # return it as-is
    if len(external_outputs) == len(model.graph.output):
        if verbose:
            print(
                "No change required for ONNX model:"
                " All nodes already exposed as outputs"
            )
        return model_path

    # 4. Do a shape and type inference pass on the model to ensure
    #    they're defined for graph outputs
    model = onnx.shape_inference.infer_shapes(model)
    # 4.5 Remove every output node for which the type or shape could not be inferred
    for i, tensor_valueproto in reversed(list(enumerate(model.graph.output))):
        if not tensor_has_valid_type(tensor_valueproto, verbose):
            del model.graph.output[i]

    if run_checker:
        try:
            onnx.checker.check_model(model)
        except onnx.checker.ValidationError as v:
            # Ignoring this specific error because the ONNX spec says a missing shape is the
            # legal way to define a tensor of unknown dimensions. Honestly, believe this is
            # a bug in the checker.
            # See https://github.com/onnx/onnx/issues/2492
            if str(v).endswith("Field 'shape' of type is required but missing."):
                if verbose:
                    print(
                        "Warning: Ignoring the following error because it is probably an "
                        "ONNX Checker error: ",
                        v,
                    )
            else:
                raise v

    if not overwrite:
        # Make a copy of the .onnx model to save it as a file
        model_path_components = model_path.rsplit(
            ".", 1
        )  # Split before and after extension
        model_path = (
            model_path_components[0] + "_exposed_nodes." + model_path_components[1]
        )
    onnx.save(model, model_path)
    return model_path


def onnxrt(
    model_path, model,
):
    """
    execute onnxruntime to validate onnx file and to extract data shapes needed
    to transform weight shapes
    """
    sess_opt = onnxruntime.SessionOptions()
    sess_opt.graph_optimization_level = (
        onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
    )

    input_shapes = {}

    input_shape = []
    input_shape.append(1)
    for _dim in range(1, len(model.graph.input[0].type.tensor_type.shape.dim)):
        input_shape.append(
            model.graph.input[0].type.tensor_type.shape.dim[_dim].dim_value
        )

    # convert model
    input_shapes[None] = input_shape
    exfile = expose_node_outputs(
        model_path, overwrite=False, run_checker=False, verbose=False
    )

    sess = onnxruntime.InferenceSession(exfile, sess_options=sess_opt)

    feeds = {}
    for input_meta in sess.get_inputs():
        # replace any symbolic dimensions
        shape = []
        for dim in input_meta.shape:
            if not dim:
                # unknown dim
                shape.append(1)
            elif type(dim) == str:  # pylint: disable=unidiomatic-typecheck
                # symbolic dim. see if we have a value otherwise use 1
                shape.append(1)
            else:
                shape.append(dim)

        if input_meta.type in float_dict:
            feeds[input_meta.name] = np.random.rand(*shape).astype(
                float_dict[input_meta.type]
            )
        elif input_meta.type in integer_dict:
            feeds[input_meta.name] = np.random.uniform(
                high=1000, size=tuple(shape)
            ).astype(integer_dict[input_meta.type])
        elif input_meta.type == "tensor(bool)":
            feeds[input_meta.name] = np.random.randint(2, size=tuple(shape)).astype(
                "bool"
            )
        else:
            eprint(
                "unsupported input type {} for input {}".format(
                    input_meta.type, input_meta.name
                )
            )
            sys.exit(-1)

    # Starting with IR4 some initializers provide default values
    # and can be overridden (available in IR4). For IR < 4 models
    # the list would be empty
    for initializer in sess.get_overridable_initializers():
        shape = [dim if dim else 1 for dim in initializer.shape]
        if initializer.type in float_dict:
            feeds[initializer.name] = np.random.rand(*shape).astype(
                float_dict[initializer.type]
            )
        elif initializer.type in integer_dict:
            feeds[initializer.name] = np.random.uniform(
                high=1000, size=tuple(shape)
            ).astype(integer_dict[initializer.type])
        elif initializer.type == "tensor(bool)":
            feeds[initializer.name] = np.random.randint(2, size=tuple(shape)).astype(
                "bool"
            )
        else:
            eprint(
                "unsupported initializer type {} for initializer {}".format(
                    initializer.type, initializer.name
                )
            )
            sys.exit(-1)

    try:
        ort_outs = sess.run(None, feeds)  # fetch all outputs
    except onnxruntime.capi.onnxruntime_pybind11_state.RuntimeException:  # pylint: disable=I1101
        eprint("Unsupported ONNX operation encountered")
        sys.exit(1)

    inps = {}

    for x, _ in enumerate(feeds):
        inps[sess.get_inputs()[x].name] = sess.get_inputs()[x]

    outs = {}
    i = 0
    for ort_out in ort_outs:
        outs[sess.get_outputs()[i].name] = ort_out
        i = i + 1

    return inps, outs


def load(  # pylint: disable=R0914
    checkpoint_file,
    cfg_layers,
    cfg,
    fc_layer,
    quantization,
    bias_quantization,
    output_shift,
    kernel_size,  # this information available in onnx model
    operator,
    verbose=False,
    no_bias=None,
    scale=None,
    generate_dequantized_onnx_file=False,
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
    do_quantize = False
    scale_factor = 1.0

    if scale is not None:
        do_quantize = True
        scale_factor = float(scale)

    # no need to scale first layer weights by 1/2
    first = False

    if generate_dequantized_onnx_file is True:
        model_path_components = checkpoint_file.rsplit(".", 1)
        output_file = model_path_components[0] + "_dq." + model_path_components[1]
        modify_weights(checkpoint_file, output_file, scale_factor, first, True)

    model = onnx.load(checkpoint_file)
    print(f"Reading {checkpoint_file} to configure network weights...")

    layers = 0
    num_conv_layers = len(quantization)
    no_bias = no_bias or []
    weights = []
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
    is_not_layer = False
    kernel_size_onnx = []
    input_dims = []
    dim = []
    matmul_out = None
    trans_perm = (0, 1, 2, 3)
    oplist = []
    t0 = 0
    t1 = 1
    t2 = 2
    t3 = 3
    squeeze = False
    sqz_dim = -1
    num_layer_ops = 0
    initializers = {t.name for t in model.graph.initializer}
    cast_out = None
    mul_out = None
    add_out = None
    cast_w = []
    cast_ref = []
    cast_w_quant = []
    cast_out_ref = []
    mul_in = None
    add_in = None
    save_shape = []
    save_perm = None
    op_list = []
    conv_relu_pool = 0

    # looking for conv1d/conv2d indications
    for x in range(cfg_layers):
        op = cfg["layers"][x].get("op", None)
        if op is None:
            op = cfg["layers"][x].get("operation", None)
        if op is None:
            op = cfg["layers"][x].get("convolution", None)
        op_list.append(op)

    _, out_dict = onnxrt(checkpoint_file, model)
    save_shape, save_perm = track_data_shape(model, out_dict)

    # find Cast/Mul/Add sequences with connected in/outs
    # and integer initializers in Cast node
    # according to quantize script, Cast initializer is quantized weights
    # and Add/Mul initializers are dequantize operands
    for _, node in enumerate(model.graph.node):
        if node.op_type == "Cast" or node.op_type == "Mul" or node.op_type == "Add":
            inputs, outputs = get_inouts(node)
            if node.op_type == "Cast":
                cast_out = outputs[0]
                for attr in node.attribute:
                    if attr.name == "to":
                        if attr.HasField("i"):
                            if attr.i == 1:
                                cast_w, _iq = process_channels(
                                    model,
                                    inputs[0],
                                    initializers,
                                    False,
                                    False,
                                    scale_factor,
                                )
                                cast_w = cast_w.astype(np.int8)
                                cast_w = np.clip(cast_w, -128, 127).round()

            if node.op_type == "Mul":
                mul_out = outputs[0]
                mul_in = inputs[0]

            if node.op_type == "Add":
                add_out = outputs[0]
                add_in = inputs[0]

                if (cast_out == mul_in) and (mul_out == add_in):
                    cast_w_quant.append(cast_w)
                    cast_out_ref.append(add_out)
                    cast_out = None
                    mul_out = None
                    add_out = None
                    cast_w = []
                    mul_in = None
                    add_in = None

    for _, node in enumerate(model.graph.node):
        oplist.append(node.op_type)

        inputs, outputs = get_inouts(node)

        if node.op_type == "Cast":
            continue

        if node.op_type == "Mul":
            continue

        if node.op_type == "Conv":
            conv_relu_pool = 1

        if node.op_type == "Relu":
            if conv_relu_pool == 1:
                conv_relu_pool = 2
            else:
                conv_relu_pool = 0

        if (node.op_type == "MaxPool") or (node.op_type == "AveragePool"):
            kernel = []
            stride = []
            for attr in node.attribute:
                if attr.name == "kernel_shape":
                    kernel_len = len(attr.ints)
                    for x in range(kernel_len):
                        kernel.append(attr.ints[x])
                if attr.name == "strides":
                    stride_len = len(attr.ints)
                    for x in range(stride_len):
                        stride.append(attr.ints[x])

            if conv_relu_pool == 2:
                conv_relu_pool = 0
                seq += 1
                layers += 1
                kernel_shape = kernel
                kernel_size_onnx.append(kernel_shape)
                new_shape = out_dict[node.output[0]].shape
                new_size1 = output_channels[-1]
                new_size0 = new_shape[1]
                input_channels.append(new_size1)
                output_channels.append(new_size0)
                weights.append(None)
                bias.append(None)
                quantization.append(None)
                continue

        if (
            node.op_type == "Conv"
            or node.op_type == "ConvTranspose"
            or node.op_type == "Gemm"
            or node.op_type == "MatMul"
            or node.op_type == "Add"
        ):
            if (
                node.op_type == "Conv"
                or node.op_type == "ConvTranspose"
                or node.op_type == "Gemm"
                or node.op_type == "MatMul"
            ):
                num_layer_ops += 1
                if node.op_type == "MatMul":
                    matmul_out = outputs[
                        0
                    ]  # reference to find following Add(matmul_out,bias)

                for inp in inputs:
                    if inp in initializers:
                        for initializer in model.graph.initializer:
                            if inp == initializer.name:
                                for d in initializer.dims:
                                    input_dims.append(d)

            if node.op_type == "Gemm" or node.op_type == "MatMul":
                kernel_shape = [1, 1]
                kernel_size_onnx.append(kernel_shape)

                if layers >= num_conv_layers:
                    continue

            if node.op_type == "Conv" or node.op_type == "ConvTranspose":
                for a in node.attribute:
                    if a.name == "kernel_shape":
                        if len(a.ints) == 1:
                            kernel_shape = [a.ints[0], 1]
                            kernel_size_onnx.append(kernel_shape)
                        else:
                            kernel_size_onnx.append(a.ints)

            for inp in inputs:
                w, internal_quantized = process_channels(
                    model, inp, initializers, do_quantize, first, scale_factor
                )
                if internal_quantized is True:
                    if len(w.shape) > 1:
                        first = False

                if w is None:
                    # tf quantize script stores weights in Cast node
                    if inp in cast_out_ref:
                        index = 0
                        for cast_ref in cast_out_ref:
                            if inp == cast_ref:
                                w = cast_w_quant[index]
                                break
                            index = index + 1

                if w is not None:
                    if (
                        node.op_type == "Gemm"
                        or node.op_type == "MatMul"
                        or node.op_type == "Add"
                    ):  # general matrix multiplication (FC layer)
                        if node.op_type == "Gemm" or node.op_type == "MatMul":
                            if fc_layer:
                                if inp == inputs[1]:  # weight
                                    assert w.min() >= -128 and w.max() <= 127
                                    fc_weights.append(w)

                                if node.op_type == "Gemm":
                                    if len(inputs) == 3:  # have optional bias input
                                        if inp == inputs[2]:  # bias
                                            assert w.min() >= -128 and w.max() <= 127
                                            fc_bias.append(w)
                                    elif inp == inputs[1]:  # add bias 'None'
                                        fc_bias.append(
                                            None
                                        )  # during weight input processing

                        if node.op_type == "Add":
                            is_not_layer = True
                            if len(oplist) > 3:
                                cst_seq = (
                                    oplist[-4] + oplist[-3] + oplist[-2]
                                    == "ConvSqueezeTranspose"
                                )

                            # TODO: Is a bias ever more than a single dimension
                            #       that needs to be transposed?

                            if inputs[0] == matmul_out or cst_seq is True:
                                if fc_layer:
                                    if inp == inputs[1]:  # bias
                                        assert w.min() >= -128 and w.max() <= 127
                                        fc_bias.append(w)
                                else:
                                    if len(bias) == seq:
                                        # remove default matmul bias entries if bias/Add detected
                                        del bias[seq - 1]
                                        del bias_min[seq - 1]
                                        del bias_max[seq - 1]
                                        del bias_keys[seq - 1]
                                        del bias_quant[seq - 1]
                                        del bias_size[seq - 1]

                                    manage_bias(
                                        w,
                                        bias,
                                        bias_min,
                                        bias_max,
                                        bias_keys,
                                        bias_quant,
                                        bias_size,
                                        param_size,
                                        bias_quantization,
                                        seq - 1,
                                        inp,
                                        param_count,
                                        do_quantize,
                                        False,
                                        scale_factor,
                                    )
                            is_not_layer = True
                            continue

                    if len(w.shape) > 1:  # not a bias
                        w_min, w_max = w.min(), w.max()

                        if quantization[seq] is not None:
                            assert w_min >= -(2 ** (quantization[seq] - 1)), print(
                                w_min
                            )
                            assert w_max < 2 ** (quantization[seq] - 1), print(w_max)
                        else:
                            if w_max > 0:
                                w_max_m = int(w_max)
                            else:
                                w_max_m = int(abs(w_max)) - 1
                            if w_min > 0:
                                w_min_m = int(w_min)
                            else:
                                w_min_m = int(abs(w_min)) - 1
                                quantization[seq] = 1 << (
                                    fls(max(fls(w_max_m), fls(w_min_m)) + 1) + 1
                                )
                            assert quantization[seq] <= 8
                        quant.append(quantization[seq])

                        w = w.astype(np.int64)

                        weight_min.append(w_min)
                        weight_max.append(w_max)

                        # TODO: Double check if we need to check conv2d if opn is known
                        # to be opn.CONVTRANSPOSE2D. We should be able to get this
                        # from the op_type Conv plus shape?
                        if operator[seq] == opn.CONVTRANSPOSE2D:
                            # For ConvTranspose2d, flip the weights as follows:
                            w = np.flip(w, axis=(2, 3))
                            w = np.transpose(w, (1, 0, 2, 3))

                        if len(w.shape) == 2:
                            if w.shape[t0] > w.shape[t1]:
                                trans_perm = (t1, t0)
                                dense_shape = (w.shape[1], w.shape[0])

                                w = w.T
                                if len(save_perm) > 0:
                                    w = np.reshape(w, save_shape)
                                    w = np.transpose(w, inv(save_perm))
                                    w = np.reshape(w, dense_shape)

                        input_channels.append(w.shape[t1])  # Input channels
                        output_channels.append(w.shape[t0])  # Output channels

                        if len(w.shape) == 2:  # MLP
                            if (
                                kernel_size_onnx[seq][0] != 1
                                or kernel_size_onnx[seq][1] != 1
                            ):
                                eprint(
                                    f"The `kernel_size` for the MLP layer {seq} should "
                                    f"be set to 1x1 instead of "
                                    f"{kernel_size[seq][0]}x{kernel_size[seq][1]}.",
                                    exit_code=None,
                                )
                                error_exit = True
                        elif len(w.shape) == 3:  # 1D
                            if (
                                kernel_size_onnx[seq][0] != w.shape[2]
                                or kernel_size_onnx[seq][1] != 1
                            ):
                                eprint(
                                    f"The `kernel_size` for the 1D layer {seq} should "
                                    f"be set to {w.shape[2]}x1 instead of "
                                    f"{kernel_size[seq][0]}x{kernel_size[seq][1]}.",
                                    exit_code=None,
                                )
                                error_exit = True
                        elif len(w.shape) == 4:  # 2D
                            if (
                                kernel_size_onnx[seq][0] != w.shape[t2]
                                or kernel_size_onnx[seq][1] != w.shape[t3]
                            ):
                                eprint(
                                    f"The `kernel_size` for the 2D layer {seq} should "
                                    f"be set to {w.shape[2]}x{w.shape[3]} instead of "
                                    f"{kernel_size[seq][0]}x{kernel_size[seq][1]}.",
                                    exit_code=None,
                                )
                                error_exit = True

                        w_count = np.prod(w.shape)
                        param_count += w_count
                        w_size = (
                            w_count * 8 + (quantization[seq] - 1)
                        ) // quantization[seq]
                        weight_size.append(w_size)
                        param_size += w_size

                        if len(w.shape) == 2:  # linear - add dummy 'channel'
                            w = np.expand_dims(w, axis=0)
                        else:  # conv1d, conv2d, ... - combine input and output channels
                            if squeeze is True:
                                w = np.reshape(w, (-1,) + w.shape[sqz_dim:])
                            else:
                                w = np.reshape(w, (-1,) + w.shape[2:])
                        weights.append(w)
                        weight_keys.append(inp)

                    if len(inputs) < 3 or (
                        ((inp == inputs[2] or cast_ref == inputs[2]) and seq in no_bias)
                    ):  # no bias input
                        bias.append(None)
                        bias_min.append(0)
                        bias_max.append(0)
                        bias_keys.append("N/A")
                        bias_quant.append(0)
                        bias_size.append(0)
                    elif inp == inputs[2]:  # bias input
                        manage_bias(
                            w,
                            bias,
                            bias_min,
                            bias_max,
                            bias_keys,
                            bias_quant,
                            bias_size,
                            param_size,
                            bias_quantization,
                            seq,
                            inp,
                            param_count,
                            do_quantize,
                            False,
                            scale_factor,
                        )

            if is_not_layer is False:
                if node.op_type != "Add":
                    seq += 1
                    layers += 1

            trans_perm = (0, 1, 2, 3)
            t0 = get_perm(0, trans_perm)
            t1 = get_perm(1, trans_perm)
            t2 = get_perm(2, trans_perm)
            t3 = get_perm(3, trans_perm)
            squeeze = False
            is_not_layer = False
        else:
            continue
        # TODO: Things to add
        # if attribute.name == 'pads':
        # if attribute.name == 'strides':

    if verbose:
        print(
            "Layer  InCh OutCh  Weights         Quant  Min Max   Size "
            "Key                                 Bias       Quant  Min Max Size Key"
        )
        for ll in range(layers):
            if ll < len(weights) and weights[ll] is not None:
                weight_shape = str(weights[ll].shape)
                if bias[ll] is not None:
                    bias_shape = str(bias[ll].shape)
                else:
                    bias_shape = "N/A"
                print(
                    f"{ll:4}: "
                    f"{input_channels[ll]:5} {output_channels[ll]:5}  "
                    f"{weight_shape:15} "
                    f"{quant[ll]:5} {weight_min[ll]:4} {weight_max[ll]:3} {weight_size[ll]:6} "
                    f"{weight_keys[ll]:35} "
                    f"{bias_shape:10} "
                    f"{bias_quant[ll]:5} {bias_min[ll]:4} {bias_max[ll]:3} {bias_size[ll]:4} "
                    f"{bias_keys[ll]:25}"
                )
        print(
            f"TOTAL: {layers} layers, {param_count:,} parameters, {param_size:,} bytes"
        )

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
            print(dim)
            print("")
            print(oplist)

    return (
        layers,
        weights,
        bias,
        output_shift,
        fc_weights,
        fc_bias,
        input_channels,
        output_channels,
    )
