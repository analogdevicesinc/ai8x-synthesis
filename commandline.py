###################################################################################################
# Copyright (C) 2018-2019 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
# Written by RM
###################################################################################################
"""
Command line parser for Tornado CNN
"""
import argparse


def get_parser():
    """
    Return an argparse parser.
    """

    parser = argparse.ArgumentParser(description="AI8X Software CNN Generator")
    parser.add_argument('--ai85', action='store_const', const=85, default=84, dest='device',
                        help="enable AI85 features (default: false)")
    parser.add_argument('--apb-base', type=lambda x: int(x, 0), metavar='N',
                        help=f"APB base address (default: device specific)")
    parser.add_argument('--autogen', default='tests', metavar='S',
                        help="directory location for autogen_list (default: 'tests'); "
                             "don't add if 'None'")
    parser.add_argument('--avg-pool-rounding', action='store_true', default=False,
                        help="round average pooling results on AI85 and AI86 (default: false)")
    parser.add_argument('--c-filename', metavar='S',
                        help="C file name base (RTL sim default: 'test' -> 'test.c', "
                             "otherwise 'main' -> 'main.c')")
    parser.add_argument('--weight-filename', metavar='S', default='weights.h',
                        help="weight header file name (default: 'weights.h')")
    parser.add_argument('--sample-filename', metavar='S', default='sampledata.h',
                        help="sample data header file name (default: 'sampledata.h')")
    parser.add_argument('-e', '--embedded-code', action='store_true', default=False,
                        help="generate embedded code for device instead of RTL simulation")
    parser.add_argument('--compact-data', action='store_true', default=False,
                        help="use memcpy() to load input data in order "
                             "to save code space in RTL simulation")
    parser.add_argument('--compact-weights', action='store_true', default=False,
                        help="use memcpy() to load weights in order "
                             "to save code space in RTL simulation")
    parser.add_argument('-f', '--fc-layer', action='store_true', default=False,
                        help="add a fully connected classification layer in software "
                             "(default: false; AI85 uses hardware)")
    parser.add_argument('--fifo', action='store_true', default=False,
                        help="use FIFOs to load streaming data (default: false)")
    parser.add_argument('-D', '--debug', action='store_true', default=False,
                        help="debug mode (default: false)")
    parser.add_argument('--debug-computation', action='store_true', default=False,
                        help="debug computation (default: false)")
    parser.add_argument('--config-file', required=True, metavar='S',
                        help="YAML configuration file containing layer configuration")
    parser.add_argument('--checkpoint-file', metavar='S',
                        help="checkpoint file containing quantized weights")
    parser.add_argument('--input-filename', default='input', metavar='S',
                        help="input .mem file name base (default: 'input' -> 'input.mem')")
    parser.add_argument('--output-filename', default='output', metavar='S',
                        help="output .mem file name base (default: 'output' -> 'output-X.mem')")
    parser.add_argument('--runtest-filename', default='run_test.sv', metavar='S',
                        help="run test file name (default: 'run_test.sv')")
    parser.add_argument('--log-filename', default='log.txt', metavar='S',
                        help="log file name (default: 'log.txt')")
    parser.add_argument('--init-tram', action='store_true', default=False,
                        help="initialize TRAM to 0 (default: false)")
    parser.add_argument('--max-proc', type=int, metavar='N',
                        help="override maximum number of processors")
    parser.add_argument('--mexpress', action='store_true', default=False,
                        help="use express kernel loading (default: false)")
    parser.add_argument('--no-error-stop', action='store_true', default=False,
                        help="do not stop on errors (default: stop)")
    parser.add_argument('--input-offset', type=lambda x: int(x, 0),
                        metavar='N',
                        help="input offset (x8 hex, defaults to 0x0000)")
    parser.add_argument('--one-shot', action='store_true', default=False,
                        help="use layer-by-layer one-shot mechanism (default: false)")
    parser.add_argument('--overlap-data', '--overwrite-ok', dest='overwrite_ok',
                        action='store_true', default=False,
                        help="allow output to overwrite input (default: warn/stop)")
    parser.add_argument('--override-start', type=lambda x: int(x, 0),
                        metavar='N',
                        help="override start value (x8 hex)")
    parser.add_argument('--override-rollover', type=lambda x: int(x, 0),
                        metavar='N',
                        help="override rollover value (x8 hex)")
    parser.add_argument('--override-delta1', type=lambda x: int(x, 0),
                        metavar='N',
                        help="override delta1 value (x8 hex)")
    parser.add_argument('--override-delta2', type=lambda x: int(x, 0),
                        metavar='N',
                        help="override delta2 value (x8 hex)")
    parser.add_argument('--queue-name', default='lowp', metavar='S',
                        help="queue name (default: 'lowp')")
    parser.add_argument('-L', '--log', action='store_true', default=False,
                        help="redirect stdout to log file (default: false)")
    parser.add_argument('--input-split', type=int, default=1, metavar='N',
                        choices=range(1, 1025),
                        help="split input into N portions (default: don't split)")
    parser.add_argument('--riscv', action='store_true', default=False,
                        help="use RISC-V processor (default: false)")
    parser.add_argument('--slow-load', type=int, metavar='N', default=0,
                        help="slow down FIFO loads (default: 0)")
    parser.add_argument('--stop-after', type=int, metavar='N',
                        help="stop after layer")
    parser.add_argument('--stop-start', action='store_true', default=False,
                        help="stop and then restart the accelerator (default: false)")
    parser.add_argument('--synthesize-input', type=int, metavar='N',
                        help="synthesize input data from first 8 lines (default: false)")
    parser.add_argument('--prefix', metavar='S', required=True,
                        help="set test name prefix")
    parser.add_argument('--test-dir', metavar='S', required=True,
                        help="set base directory name for auto-filing .mem files")
    parser.add_argument('--top-level', default=None, metavar='S',
                        help="top level name instead of block mode (default: None)")
    parser.add_argument('--timeout', type=int, metavar='N',
                        help="set RTL sim timeout (units of 1ms, default based on test)")
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help="verbose output (default: false)")
    parser.add_argument('--verify-writes', action='store_true', default=False,
                        help="verify write operations (toplevel only, default: false)")
    parser.add_argument('--verify-kernels', action='store_true', default=False,
                        help="verify kernels (toplevel only, default: false)")
    parser.add_argument('--write-zero-registers', action='store_true', default=False,
                        help="write registers even if the value is zero (default: do not write)")
    parser.add_argument('--zero-sram', action='store_true', default=False,
                        help="zero memories (default: false)")
    parser.add_argument('--zero-unused', action='store_true', default=False,
                        help="zero unused registers (default: do not touch)")
    parser.add_argument('--cmsis-software-nn', action='store_true', default=False,
                        help="create code for an Arm CMSIS NN software network instead")
    parser.add_argument('--mlator', action='store_true', default=False,
                        help="use hardware to swap output bytes (default: false)")
    parser.add_argument('--ready-sel', type=int, metavar='N',
                        help="specify memory waitstates")
    parser.add_argument('--ready-sel-fifo', type=int, metavar='N',
                        help="specify FIFO waitstates")
    args = parser.parse_args()

    if not args.c_filename:
        args.c_filename = 'main' if args.embedded_code else 'test'

    return args
