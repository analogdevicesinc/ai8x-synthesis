###################################################################################################
# Copyright (C) Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
###################################################################################################
"""
Generic backend for code generation
"""


class Backend:
    """
    Abstract class for all backends
    """
    def create_net(self) -> str:  # pylint: disable=no-self-use
        """
        Contruct a CNN and create C code for a given device
        """
        return ''
