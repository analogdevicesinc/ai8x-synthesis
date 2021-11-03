#!/usr/bin/env python3
###################################################################################################
# Copyright (C) 2020-2021 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
###################################################################################################
"""
Embedded network and simulation test generator program for Tornado CNN
"""
import signal
import sys

from izer.izer import main


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
