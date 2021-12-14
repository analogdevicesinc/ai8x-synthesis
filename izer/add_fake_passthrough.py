###################################################################################################
# Copyright (C) 2021 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
###################################################################################################
"""
Command line tool to add passthrough layer to a quantized model as identity Conv2D kernels.
"""
import argparse
import copy
from collections import OrderedDict

import torch
from torch import nn


def parse_arguments():
    """Parses command line arguments"""
    parser = argparse.ArgumentParser(description="Fake Passthrough Layer Insertion")
    parser.add_argument('--input-checkpoint-path', metavar='S', required=True,
                        help="path to checkpoint file")
    parser.add_argument('--output-checkpoint-path', metavar='S', required=True,
                        help="path to checkpoint file")
    parser.add_argument('--layer-name', metavar='S', required=True,
                        help='name of the added passtrhough layer')
    parser.add_argument('--layer-depth', type=int, required=True,
                        help='depth of the passthrough layer')
    parser.add_argument('--layer-name-after-pt', metavar='S', required=True,
                        help='name of the layer just after the passthrough layer is added')

    args = parser.parse_args()
    return args


def passthrough_faker(n_channels):
    """Creates passthrough layer"""
    a = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=1, bias=False)
    a.weight.data = torch.zeros_like(a.weight.data)
    for i in range(a.weight.data.shape[0]):
        a.weight.data[i, i, :, :] = 64
    return a


def main():
    """Main function to add passthrough layer"""
    args = parse_arguments()
    device = torch.device('cpu')

    checkpoint = torch.load(args.input_checkpoint_path)
    passthrough_kernel = passthrough_faker(args.layer_depth)

    new_checkpoint = copy.deepcopy(checkpoint)

    # remove `module.` prefix from the state dictionary keys if model is trained with GPU
    # (see:https://discuss.pytorch.org/t/prefix-parameter-names-in-saved-model-if-trained-by-multi-
    # gpu/494)
    new_state_dict = OrderedDict()
    for k, v in new_checkpoint['state_dict'].items():
        name = k.replace("module.", '')
        new_state_dict[name] = v

    new_state_dict[f'{args.layer_name}.output_shift'] = torch.Tensor([1.]).to(device)
    new_state_dict[f'{args.layer_name}.weight_bits'] = torch.Tensor([8.]).to(device)
    new_state_dict[f'{args.layer_name}.bias_bits'] = torch.Tensor([8.]).to(device)
    new_state_dict[f'{args.layer_name}.quantize_activation'] = torch.Tensor([1.]).to(device)
    new_state_dict[f'{args.layer_name}.adjust_output_shift'] = torch.Tensor([0.]).to(device)
    new_state_dict[f'{args.layer_name}.shift_quantile'] = torch.Tensor([1.]).to(device)
    new_state_dict[f'{args.layer_name}.op.weight'] = passthrough_kernel.weight.data.to(device)

    move_layer = False
    for key in list(new_state_dict.keys()):
        if not move_layer and key.startswith(args.layer_name_after_pt):
            move_layer = True

        if move_layer and key.startswith(args.layer_name):
            move_layer = False

        if move_layer:
            new_state_dict.move_to_end(key)

    new_checkpoint['state_dict'] = new_state_dict
    torch.save(new_checkpoint, args.output_checkpoint_path)


if __name__ == '__main__':
    main()
