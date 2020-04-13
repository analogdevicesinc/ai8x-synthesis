###################################################################################################
# Copyright (C) 2020 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
# Written by RM
###################################################################################################
"""
Print error message to stderr, and stdout as well if needed
"""
import sys


def eprint(*args, **kwargs):
    """
    Print message to stderr, and stdout as well IF stdout was overridden.
    """

    if sys.stdout != sys.__stdout__:
        print(*args, **kwargs)

    print(*args, file=sys.stderr, **kwargs)
