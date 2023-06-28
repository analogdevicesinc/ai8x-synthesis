###################################################################################################
# Copyright (C) 2022 Maxim Integrated Products Inc. All Rights Reserved.
#
# Maxim Integrated Products Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
###################################################################################################
"""
Layer names.
"""
from typing import List, Optional

from . import state
from .eprint import eprint


def find_layer(
    all_names: List,  # contains "layer_name"s as first element, "data_buffer_name" as second
    sequence: int,
    name: str,
    keyword: str,
    error: bool = True,
) -> Optional[int]:
    """
    Find layer number given a layer name.
    """
    name = name.lower()
    if name == 'input':
        return -1
    layer_names = all_names[0]
    for ll, e in enumerate(layer_names):
        if e is not None and e.lower() == name:
            return ll
    if name == all_names[1]:
        return -2   # data buffer
    if error:
        eprint(f'Could not find the `{keyword}` layer name `{name}` in layer sequence '
               f'{sequence} of the YAML configuration file.')
    return None


def layer_str(
    ll: int,
) -> str:
    """
    Convert a layer number to a layer name.
    """
    if ll == -1:
        return 'input'
    name = state.layer_name[ll]
    if name is not None:
        return f'{ll} ({name})'
    return str(ll)


def layer_pfx(
    ll: int,
) -> str:
    """
    Convert a layer number to a layer name prefixed by "Layer " and followed by ":".
    """
    return f'Layer {layer_str(ll)}: '
