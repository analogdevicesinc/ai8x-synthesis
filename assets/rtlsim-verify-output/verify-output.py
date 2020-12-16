#!/usr/bin/env python3.7
###################################################################################################
# Copyright (C) Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
###################################################################################################
"""
Compare the data memory contents from output of a simulation to the expected data.
Run from the simulation directory.
"""
import os
import sys

if len(sys.argv) > 2:
    print('ERROR: Unknown arguments', sys.argv, file=sys.stderr)
    sys.exit(-1)

# Got target directory on command line
if len(sys.argv) == 2:
    if not os.path.isdir(sys.argv[1]):
        print('ERROR: Argument', sys.argv[1], 'is not a directory!', file=sys.stderr)
        sys.exit(-1)
    os.chdir(sys.argv[1])

MAX_MISMATCH = 10

# Check whether 'data-output' and 'data-expected' exist.
if not os.path.isdir('data-output') or not os.path.isdir('data-expected'):
    print('ERROR: data-output/ or data-expected/ does not exist!', file=sys.stderr)
    sys.exit(-1)

failures = 0
matches = 0

# Go through all the files in 'data-expected' and read them. For each, make sure there is a
# corresponding file in 'data-output'. Open that file and make sure that the information matches.
for _, _, fnames in sorted(os.walk('data-expected')):
    for fname in sorted(fnames):
        outname = fname.replace('DRAM_', 'DRAM_out_')
        if not os.path.isfile(os.path.join('data-output', outname)):
            print(f'ERROR: data-output/{outname} does not exist!', file=sys.stderr)
            sys.exit(-2)

        with open(os.path.join('data-output', outname)) as f:
            data = f.readlines()

        with open(os.path.join('data-expected', fname)) as f:
            expected = f.readlines()

        for e in expected:
            addr, val = e.split(' ')
            if addr[0] != '@':
                print(f'ERROR: Malformed line {e.strip()} in file data-expected/{fname}!',
                      file=sys.stderr)
                sys.exit(-3)

            try:
                addr = int(addr[1:], base=16)
                val = val.strip().lower()
                mask = ''
                for _, m in enumerate(val):
                    mask += '0' if m == 'x' else 'f'
                mask = int(mask, base=16)
                val = int(val.replace('x', '0'), base=16)
            except ValueError:
                print(f'ERROR: Malformed line {e.strip()} in file data-expected/{fname}!',
                      file=sys.stderr)
                sys.exit(-3)

            if addr > len(data):
                print(f'ERROR: Address from {e.strip()} not present in '
                      f'file data-output/{outname}!', file=sys.stderr)
                failures += 1
            else:
                result = data[addr].strip().lower()
                if result == 'x' * len(result):
                    print(f'ERROR: Output is {result} at address {addr:04x} in '
                          f'file data-output/{outname}!', file=sys.stderr)
                    failures += 1
                else:
                    try:
                        result0 = int(result.replace('x', '0'), base=16)
                        resultf = int(result.replace('x', 'f'), base=16)
                    except ValueError:
                        print(f'ERROR: Malformed line {addr}: {data[addr].strip()} in '
                              f'file data-output/{fname}!', file=sys.stderr)
                        sys.exit(-3)

                    # print(f'Found address {addr:04x}, val {val:08x}, comp {result}')
                    if result0 & mask != val & mask or resultf & mask != val & mask:
                        if failures == 0:
                            print(f'Before this failure, {matches} values were correct.',
                                  file=sys.stderr)
                        print(f'ERROR: Data mismatch at address {addr:04x} in '
                              f'file data-output/{outname}. Expected: {val:04x}, '
                              f'got {result} (mask {mask:08x})!', file=sys.stderr)
                        failures += 1
                    else:
                        matches += 1

            if failures > MAX_MISMATCH:
                print(f'ERROR: Exceeding maximum compare failures ({MAX_MISMATCH}), exiting!',
                      file=sys.stderr)
                sys.exit(failures)

if failures == 0:
    print('SUCCESS:', matches, 'data word matches.', file=sys.stderr)

# Return success (0) or number of failures (when < MAX_MISMATCH):
sys.exit(failures)
