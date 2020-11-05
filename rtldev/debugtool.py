#!/usr/bin/env python3
###################################################################################################
# Copyright (C) Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
# Written by RM
###################################################################################################
"""
Debug RTL simulation output streams.

The RTL must have datalog enabled. Once the simulation is complete, run this tool as follows:
    debugtool.py test_*.txt
It will create .mem files for each layers. The output*.mem files should match the files created
when running ai8xize.py with the --intermediate-data argument. Further debug will most likely
also need the pooled/unpooled data in the log.txt file or in the CSV files created by ai8xize.py
(--log-pooling argument).
"""

import argparse
import re
from glob import glob
from typing import List

DATA_XX = '00'
write_access = None
read_access = None
stream = 0


def parse(group, line):
    """
    Parse a single `line` from the test*.txt file for `group`.
    """
    global write_access, read_access, stream  # pylint: disable=global-statement

    ll = line.strip().replace(' @ ', ',').replace(' = ', ',').replace('ps', '').split(sep=',')
    if group == 0 and ll[0].startswith('Write Pointer/Write Data'):
        time = float(ll[1])
        addr = int(ll[2], 16)
        data = int(ll[3].replace('xx', DATA_XX), 16)
        write_access.append((stream, time, addr, data))
    elif ll[0].startswith('Read Pointer/Read Data'):
        time = float(ll[1])
        addr = int(ll[2], 16)
        # addr += group * x
        if ll[3] != 'xxxxxxxx':
            data0 = int(ll[3].replace('xx', DATA_XX), 16)
        else:
            data0 = -1
        if ll[4] != 'xxxxxxxx':
            data1 = int(ll[4].replace('xx', DATA_XX), 16)
        else:
            data1 = -1
        if ll[5] != 'xxxxxxxx':
            data2 = int(ll[5].replace('xx', DATA_XX), 16)
        else:
            data2 = -1
        if ll[6] != 'xxxxxxxx':
            data3 = int(ll[6].replace('xx', DATA_XX), 16)
        else:
            data3 = -1
        read_access.append((stream, time, addr, data0, data1, data2, data3))
    elif ll[0].startswith('Escalating to Stream'):
        stream = int(ll[0].split(sep='     ')[1])
    elif ll[0].startswith('De-escalating to Stream'):
        stream = int(ll[0].split(sep='  ')[1])


def main(files):
    """
    Work on all `files` specified on the command line.
    """
    global write_access, read_access, stream  # pylint: disable=global-statement

    write_access = []
    read_access = []
    readwrite = re.compile('test_[0-9]+\\.txt')

    for fp in files:
        if readwrite.match(fp):
            group = int(fp.split(sep='_')[1].split(sep='.')[0])
            with open(fp, mode='r') as f:
                stream = 0
                line = f.readline()
                while line != '':
                    parse(group, line)
                    line = f.readline()

    write_access.sort()  # Sort by stream, time, then address
    read_access.sort()

    max_stream = write_access[-1][0]

    for i in range(1, max_stream+1):
        filtered = [e for e in write_access if e[0] == i]
        with open(f'output_{i-1}.mem', mode='w') as f:
            for _, e in enumerate(filtered):
                f.write(f'{e[3]:08x}\n')

    # for _, e in enumerate(write_access):
    #     print(f'{e[0]},{e[2]:08x},{e[3]:08x}')

    # for _, e in enumerate(read_access):
    #     print(f'{e[0]},{e[2]:08x},{e[3]:08x},{e[4]:08x},{e[5]:08x},{e[6]:08x}')

    max_stream = read_access[-1][0]

    filtered = [e for e in read_access if e[0] == 1]
    with open('input_0.mem', mode='w') as f:
        for _, e in enumerate(filtered):
            f.write(f'{e[3]:08x}\n')

    for i in range(2, max_stream+1):
        filtered = [e for e in read_access if e[0] == i]
        with open(f'input_{i-1}.mem', mode='w') as f:
            for _, e in enumerate(filtered):
                f.write(f'{e[3]:08x}\n{e[4]:08x}\n{e[5]:08x}\n{e[6]:08x}\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file_names", nargs='*')

    args = parser.parse_args()
    fn: List[str] = []

    for arg in args.file_names:
        fn += glob(arg)

    main(fn)
