###################################################################################################
# Copyright (C) 2019-2023 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
###################################################################################################
"""
Load contents of a checkpoint files and save them in a quantized format.
"""
import argparse
from functools import partial

import torch

from . import tornadocnn as tc
from . import yamlcfg
from .devices import device
from .eprint import eprint, wprint

CONV_SCALE_BITS = 8
CONV_DEFAULT_WEIGHT_BITS = 8
CONV_DEFAULT_BIAS_BITS = 8
DEFAULT_SCALE = .85
DEFAULT_STDDEV = 2.0


def unwrap(x):
    """
    If `x` is a tensor, return the underlying numpy data, else return `x`.
    """
    return x.numpy() if isinstance(x, torch.Tensor) else x


def convert_checkpoint(input_file, output_file, arguments):
    """
    Convert checkpoint file or dump parameters for C code
    """
    # Load configuration file
    if arguments.config_file:
        _, _, params = yamlcfg.parse(arguments.config_file)
    else:
        params = None

    print("Converting checkpoint file", input_file, "to", output_file)
    checkpoint = torch.load(input_file, map_location='cpu')

    if 'state_dict' not in checkpoint:
        eprint("No `state_dict` in checkpoint file.")

    checkpoint_state = checkpoint['state_dict']
    compression_sched = checkpoint['compression_sched'] \
        if 'compression_sched' in checkpoint else None

    if arguments.verbose:
        print(f"\nModel keys (state_dict):\n{', '.join(list(checkpoint_state.keys()))}")

    new_checkpoint_state = checkpoint_state.copy()
    new_compression_sched = compression_sched.copy() if compression_sched is not None else {}
    new_masks_dict = new_compression_sched['masks_dict'] \
        if 'masks_dict' in new_compression_sched else None

    def avg_max(t):
        dim = 0
        view_dims = [t.shape[i] for i in range(dim + 1)] + [-1]
        tv = t.view(*view_dims)
        avg_min, avg_max = tv.min(dim=-1)[0], tv.max(dim=-1)[0]
        return torch.max(avg_min.mean().abs_(), avg_max.mean().abs_())

    def max_max(t):
        return torch.max(t.min().abs_(), t.max().abs_())

    def mean_n_stds_max_abs(t, n_stds=1):
        if n_stds <= 0:
            raise ValueError(f'n_stds must be > 0, got {n_stds}')
        mean, std = t.mean(), t.std()
        min_val = torch.max(t.min(), mean - n_stds * std)
        max_val = torch.min(t.max(), mean + n_stds * std)
        return torch.max(min_val.abs_(), max_val.abs_())

    def get_const(_):
        return arguments.scale

    def get_max_bit_shift(t, clamp_bits, shift_quantile, return_bit_shift=False):
        float_scale = 1.0 / torch.quantile(t.abs(), shift_quantile)
        bit_shift = torch.floor(torch.log2(float_scale)) \
            .clamp(min=-7.-clamp_bits, max=23.-clamp_bits)
        if return_bit_shift:
            return bit_shift

        return torch.pow(2., bit_shift)

    # If not using quantization-aware training (QAT),
    # scale to our fixed point representation using any of four methods
    # The 'magic constant' seems to work best for SCALE
    if 'extras' not in checkpoint:
        wprint("No `extras` in checkpoint file.")
        checkpoint['extras'] = {}
    if arguments.clip_mode is not None:
        if arguments.clip_mode == 'STDDEV':
            sat_fn = partial(mean_n_stds_max_abs, n_stds=arguments.stddev)
            checkpoint['extras']['clipping_method'] = 'STDDEV'
            checkpoint['extras']['clipping_nstds'] = arguments.stddev
        elif arguments.clip_mode == 'MAX':
            sat_fn = max_max
            checkpoint['extras']['clipping_method'] = 'MAX'
        elif arguments.clip_mode == 'AVGMAX':
            sat_fn = avg_max
            checkpoint['extras']['clipping_method'] = 'AVGMAX'
        else:
            sat_fn = get_const
            checkpoint['extras']['clipping_method'] = 'SCALE'
            checkpoint['extras']['clipping_scale'] = arguments.scale
    else:
        sat_fn = get_max_bit_shift
        checkpoint['extras']['clipping_method'] = 'MAX_BIT_SHIFT'

    layers = 0
    num_layers = len(params['quantization']) if params else None
    for k in checkpoint_state.keys():
        param_levels = k.rsplit(sep='.', maxsplit=2)
        if len(param_levels) == 3:
            layer, operation, parameter = param_levels[0], param_levels[1], param_levels[2]
        elif len(param_levels) == 2:
            layer, operation, parameter = param_levels[0], None, param_levels[1]
        else:
            continue

        if parameter in ['w_zero_point', 'b_zero_point']:
            if checkpoint_state[k].nonzero().numel() != 0:
                raise RuntimeError(f"\nParameter {k} is not zero.")
            del new_checkpoint_state[k]
        elif parameter == 'weight':
            if num_layers and layers >= num_layers:
                continue

            # Determine how many bits we have for the weights in this layer
            clamp_bits = None

            # First priority: Override via YAML specification
            if params is not None and 'quantization' in params:
                clamp_bits = params['quantization'][layers]

            # Second priority: Saved in checkpoint file
            if clamp_bits is None:
                weight_bits_name = '.'.join([layer, 'weight_bits'])
                if weight_bits_name in checkpoint_state:
                    layer_weight_bits = int(unwrap(checkpoint_state[weight_bits_name]))
                    if layer_weight_bits != 0:
                        clamp_bits = layer_weight_bits

            # Third priority: --qat-weight-bits or default
            if clamp_bits is None:
                if arguments.qat_weight_bits is not None:
                    clamp_bits = arguments.qat_weight_bits
                else:
                    clamp_bits = tc.dev.DEFAULT_WEIGHT_BITS  # Default to 8 bits

            bias_name = '.'.join([layer, operation, 'bias'])
            params_r = torch.flatten(checkpoint_state[k])
            if sat_fn is get_max_bit_shift:
                if bias_name in checkpoint_state:
                    weight_r = torch.flatten(checkpoint_state[k])
                    bias_r = torch.flatten(checkpoint_state[bias_name])
                    params_r = torch.cat((weight_r, bias_r))

                shift_quantile_name = '.'.join([layer, 'shift_quantile'])
                shift_quantile = 1.0
                if shift_quantile_name in checkpoint_state:
                    shift_quantile = checkpoint_state[shift_quantile_name]

                distribution_factor = get_max_bit_shift(params_r, clamp_bits, shift_quantile)
                factor = 2**(clamp_bits-1) * distribution_factor
            else:
                factor = 2**(clamp_bits-1) * sat_fn(checkpoint_state[k])

            if arguments.verbose:
                print(k, 'avg_max:', unwrap(avg_max(checkpoint_state[k])),
                      'max:', unwrap(max_max(checkpoint_state[k])),
                      'mean:', unwrap(checkpoint_state[k].mean()),
                      'factor:', unwrap(factor),
                      'bits:', clamp_bits)
            weights = factor * checkpoint_state[k]

            # Ensure it fits and is an integer
            weights = weights.add(.5).floor().clamp(min=-(2**(clamp_bits-1)),
                                                    max=2**(clamp_bits-1)-1)

            # Store modified weight back into model
            new_checkpoint_state[k] = weights

            # Set weight_bits
            weight_bits_name = '.'.join([layer, 'weight_bits'])
            if weight_bits_name not in new_checkpoint_state:
                new_checkpoint_state[weight_bits_name] = \
                    torch.Tensor([CONV_DEFAULT_WEIGHT_BITS])
                if new_masks_dict is not None:
                    new_masks_dict[weight_bits_name] = torch.Tensor([CONV_DEFAULT_WEIGHT_BITS])
            elif int(unwrap(new_checkpoint_state[weight_bits_name])) == 0:
                new_checkpoint_state[weight_bits_name] = \
                    torch.Tensor([CONV_DEFAULT_WEIGHT_BITS])

            # Is there a bias for this layer? Use the same factor as for weights.
            if bias_name in checkpoint_state:
                bias_bits_name = '.'.join([layer, 'bias_bits'])
                if arguments.verbose:
                    print(bias_name, 'avg_max:', unwrap(avg_max(checkpoint_state[bias_name])),
                          'max:', unwrap(max_max(checkpoint_state[bias_name])),
                          'mean:', unwrap(checkpoint_state[bias_name].mean()),
                          'factor:', unwrap(factor),
                          'bits:', clamp_bits)

                bias = checkpoint_state[bias_name]
                if distribution_factor:
                    bias = bias * distribution_factor

                bias = (2**(clamp_bits-1)*bias).add(0.5).floor(). \
                    clamp(min=-(2**(clamp_bits-1)),
                          max=2**(clamp_bits-1)-1)

                # Since the device multiplies the biases with 8 bits by default, the range
                # of bias needs to be shrunk according to the weight bits.
                bias *= 2**(clamp_bits-8)

                # Save conv biases so PyTorch can still use them to run a model. This needs
                # to be reversed before loading the weights into the hardware.
                # When multiplying data with weights, 1.0 * 1.0 corresponds to 128 * 128 and
                # we divide the output by 128 to compensate. The bias therefore needs to be
                # multiplied by 128. This depends on the data width, not the weight width,
                # and is therefore always 128.
                bias *= 2**(tc.dev.ACTIVATION_BITS-1)

                # Store modified bias back into model
                new_checkpoint_state[bias_name] = bias

                # Set bias_bits to default
                new_checkpoint_state[bias_bits_name] = \
                    torch.Tensor([clamp_bits])

            # Set output shift
            if arguments.clip_mode is None:
                out_shift_name = '.'.join([layer, 'output_shift'])
                out_shift = torch.Tensor([-1 * get_max_bit_shift(params_r, clamp_bits,
                                                                 shift_quantile, True)])
                new_checkpoint_state[out_shift_name] = out_shift
                if new_masks_dict is not None:
                    new_masks_dict[out_shift_name] = out_shift

            layers += 1
        elif parameter in ['base_b_q']:
            del new_checkpoint_state[k]
        elif parameter == 'adjust_output_shift':
            new_checkpoint_state[k] = torch.Tensor([0.])
        elif parameter == 'quantize_activation':
            new_checkpoint_state[k] = torch.Tensor([1.])

    checkpoint['state_dict'] = new_checkpoint_state
    if compression_sched is not None and new_masks_dict is not None:
        new_compression_sched['masks_dict'] = new_masks_dict
        checkpoint['compression_sched'] = new_compression_sched
    torch.save(checkpoint, output_file)


def main():
    """
    Command-line wrapper for quantization script.
    """
    parser = argparse.ArgumentParser(description='Checkpoint to MAX78000 Quantization')
    parser.add_argument('input', help='path to the checkpoint file')
    parser.add_argument('output', help='path to the output file')
    parser.add_argument('-c', '--config-file', metavar='S',
                        help="optional YAML configuration file containing layer configuration")
    parser.add_argument('--device', type=device, metavar='N', help="set device", required=True)
    parser.add_argument('--clip-method', default=None, dest='clip_mode',
                        choices=['AVGMAX', 'MAX', 'STDDEV', 'SCALE'],
                        help='disable quantization-aware training (QAT) information and choose '
                             'saturation clipping method')
    parser.add_argument('--qat-weight-bits', type=int,
                        help='override number of weight bits used in QAT')
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='verbose mode')
    parser.add_argument('--scale', type=float,
                        help='set the scale value for the SCALE method (default: magic '
                             f'{DEFAULT_SCALE:.2f})')
    parser.add_argument('--stddev', type=float,
                        help='set the number of standard deviations for the STDDEV method '
                             f'(default: {DEFAULT_STDDEV:.2f})')
    args = parser.parse_args()

    if args.clip_mode == 'SCALE' and not args.scale:
        wprint(f'Using the default scale factor of {DEFAULT_SCALE:.2f}.\n')
        args.scale = DEFAULT_SCALE
    if args.clip_mode == 'STDDEV' and not args.stddev:
        wprint(f'Using the default number of standard deviations of {DEFAULT_STDDEV:.2f}.\n')
        args.stddev = DEFAULT_STDDEV
    tc.dev = tc.get_device(args.device)

    convert_checkpoint(args.input, args.output, args)
