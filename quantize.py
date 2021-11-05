#!/usr/bin/env python3
###################################################################################################
# Copyright (C) 2020-2021 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
###################################################################################################
"""
Load contents of a checkpoint files and save them in a quantized format.
"""
import signal
import sys

from izer.quantize import main


def signal_handler(
        _signal,
        _frame,
):
    """
    Ctrl+C handler
    """
    sys.exit(0)


if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    main()
