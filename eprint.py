###################################################################################################
# Copyright (C) Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
###################################################################################################
"""
Print error message to stderr, and stdout as well if needed
"""
import sys

import colorama


def eprint(*args, error=True, prefix=True, exit_code=1, **kwargs):
    """
    Print message to stderr, and stdout as well IF stdout was overridden.
    Add a `prefix` if set (and `error` chooses which).
    """
    if prefix:
        pfx = 'ERROR:' if error else 'WARNING:'

        if sys.stdout != sys.__stdout__:
            print(pfx, *args, **kwargs)

        ansi_on = colorama.Fore.RED if error else colorama.Fore.YELLOW
        pfx = ansi_on + pfx + colorama.Style.RESET_ALL

        print(pfx, *args, file=sys.stderr, **kwargs)
    else:
        if sys.stdout != sys.__stdout__:
            print(*args, **kwargs)

        print(*args, file=sys.stderr, **kwargs)

    if error and exit_code:
        sys.exit(error)


def wprint(*args, **kwargs):
    """
    Print message to stderr, and stdout as well IF stdout was overridden.
    Add a WARNING: prefix.
    """
    eprint(*args, error=False, **kwargs)


def eprint_noprefix(*args, **kwargs):
    """
    Print message to stderr, and stdout as well IF stdout was overridden.
    """
    eprint(*args, prefix=False, exit_code=None, **kwargs)
