#!/usr/bin/env python3.7
###################################################################################################
# Copyright (C) 2020-2022 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
###################################################################################################
"""
Compare the data memory contents from output of a simulation to the expected data.
Run from the simulation directory.
"""
import logging
import os
import sys

RUNTEST_FILE = 'run_test.log.fail'

stdout_handler = logging.StreamHandler(sys.stdout)
handlers = [stdout_handler]

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s',
    handlers=handlers,
)
log = logging.getLogger(sys.argv[0])

if len(sys.argv) > 2:
    log.error('Unknown arguments %s', sys.argv)
    sys.exit(-1)

# Got target directory on command line
if len(sys.argv) == 2:
    if not os.path.isdir(sys.argv[1]):
        log.error('Argument %s is not a directory!', sys.argv[1])
        sys.exit(-1)
    os.chdir(sys.argv[1])

MAX_MISMATCH = 10

# Check whether 'data-output' and 'data-expected' exist.
if not os.path.isdir('data-output') or not os.path.isdir('data-expected'):
    log.error('data-output/ or data-expected/ does not exist!')
    sys.exit(-1)

failures = 0
matches = 0

# Go through all the files in 'data-expected' and read them. For each, make sure there is a
# corresponding file in 'data-output'. Open that file and make sure that the information matches.
for _, _, fnames in sorted(os.walk('data-expected')):  # type: ignore
    for fname in sorted(fnames):
        if not fname.startswith('DRAM_'):
            continue
        outname = fname.replace('DRAM_', 'DRAM_out_')
        if not os.path.isfile(os.path.join('data-output', outname)):
            log.error('data-output/%s does not exist!', outname)
            sys.exit(-2)

        with open(os.path.join('data-output', outname), encoding='utf-8') as f:
            data = f.readlines()

        with open(os.path.join('data-expected', fname), encoding='utf-8') as f:
            expected = f.readlines()

        for e in expected:
            addr, val = e.split(' ')
            if addr[0] != '@':
                log.error('Malformed line %s in file data-expected/%s!', e.strip(), fname)
                sys.exit(-3)

            try:
                addr = int(addr[1:], base=16)  # type: ignore
                val = val.strip().lower()
                mask = ''
                for m in val:  # type: ignore
                    mask += '0' if m == 'x' else 'f'
                mask = int(mask, base=16)  # type: ignore
                val = int(val.replace('x', '0'), base=16)  # type: ignore
            except ValueError:
                log.error('Malformed line %s in file data-expected/%s!', e.strip(), fname)
                sys.exit(-3)

            if addr > len(data):  # type: ignore
                log.error('Address from %s not present in file data-output/%s!',
                          e.strip(), outname)
                failures += 1
            else:
                result = data[addr].strip().lower()  # type: ignore
                if result == 'x' * len(result):
                    log.error('Output is %s at address %04x in file data-output/%s!',
                              result, addr, outname)
                    failures += 1
                else:
                    try:
                        result0 = int(result.replace('x', '0'), base=16)
                        resultf = int(result.replace('x', 'f'), base=16)
                    except ValueError:
                        log.error('Malformed line %04x: %s in file data-output/%s!',
                                  addr, data[addr].strip(), fname)  # type: ignore
                        sys.exit(-3)

                    log.debug('Found address %04x, val %08x, comp %s', addr, val, result)
                    if (
                        result0 & mask != val & mask  # type: ignore
                        or resultf & mask != val & mask  # type: ignore
                       ):
                        log.error('Data mismatch at address %04x in file data-output/%s. '
                                  'Expected: %08x, got %s (mask %08x)!',
                                  addr, outname, val, result, mask)
                        if failures == 0:
                            log.error('Before this failure, %d values were correct.', matches)
                        failures += 1
                    else:
                        matches += 1

            if failures > MAX_MISMATCH:
                log.error('Exceeding maximum compare failures (%d), exiting!', MAX_MISMATCH)
                sys.exit(failures)

if failures == 0:
    log.info('%d successful data word matches, no failures.', matches)

if not os.path.isfile(RUNTEST_FILE):
    log.error('No %s, exiting!', RUNTEST_FILE)
    sys.exit(-1)

failures = 0
matches = 0

# Check whether 'latency.txt' exists and get cycle count
if os.path.isfile('latency.txt'):
    with open('latency.txt', encoding='utf-8') as f:
        cycles = int(f.read().split()[0])

    with open(RUNTEST_FILE, encoding='utf-8') as f:
        data = f.readlines()

    actual_cycles = -1
    for d in data:
        if d.startswith("CNN Cycles = "):
            actual_cycles = int(d[13:].split()[0])
            break

    if actual_cycles == -1:
        log.error('Did not find "CNN Cycles = " in %s!', RUNTEST_FILE)
        sys.exit(-1)

    if cycles == actual_cycles:
        log.debug('Cycle counts match (%d)', cycles)
    else:
        log.error('Cycle count mismatch! Expected: %d, simulated: %d.', cycles, actual_cycles)
        failures += 1

# Return success (0) or number of failures (when < MAX_MISMATCH):
sys.exit(failures)
