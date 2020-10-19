###################################################################################################
# Copyright (C) Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
###################################################################################################
"""
Command line parser for Tornado CNN
"""
import argparse
import camera
import devices
from devices import device


def get_parser():
    """
    Return an argparse parser.
    """

    parser = argparse.ArgumentParser(description="AI8X CNN Generator")

    # Device selection
    group = parser.add_argument_group('Device selection')
    mgroup = group.add_mutually_exclusive_group()
    mgroup.add_argument('--ai85', action='store_const', const=85, dest='device',
                        help="enable AI85 features (default: AI84)")
    mgroup.add_argument('--ai87', action='store_const', const=87, dest='device',
                        help="enable AI87 features (default: AI84)")
    mgroup.add_argument('--device', type=device, metavar='N',
                        help="set device (default: 84)")
    mgroup.add_argument('--cmsis-software-nn', action='store_const',
                        const=devices.CMSISNN, dest='device',
                        help="create code for an Arm CMSIS NN software network")

    # Hardware features
    group = parser.add_argument_group('Hardware features')
    group.add_argument('--avg-pool-rounding', action='store_true', default=False,
                       help="round average pooling results on AI85 and up (default: false)")
    group.add_argument('--simple1b', action='store_true', default=False,
                       help="use simple XOR instead of 1-bit multiplication (default: false)")

    # Embedded code
    group = parser.add_argument_group('Embedded code')
    mgroup = group.add_mutually_exclusive_group()
    mgroup.add_argument('-e', '--embedded-code', action='store_true',
                        help="generate embedded code for device (default)")
    mgroup.add_argument('--rtl', '--rtl-sim', action='store_false', dest='embedded_code',
                        help="generate RTL sim code instead of embedded code (default: false)")
    group.add_argument('--config-file', required=True, metavar='S',
                       help="YAML configuration file containing layer configuration")
    group.add_argument('--checkpoint-file', metavar='S',
                       help="checkpoint file containing quantized weights")
    group.add_argument('--board-name', metavar='S', default='EvKit_V1',
                       help="set board name (default: EvKit_V1)")
    group.add_argument('--display-checkpoint', action='store_true', default=False,
                       help="show parsed checkpoint data")
    group.add_argument('--prefix', metavar='S', required=True,
                       help="set test name prefix")

    # Code generation
    group = parser.add_argument_group('Code generation')
    group.add_argument('--compact-data', action='store_true', default=False,
                       help="use memcpy() to load input data in order to save code space")
    group.add_argument('--compact-weights', action='store_true', default=False,
                       help="use memcpy() to load weights in order to save code space")
    group.add_argument('--mexpress', action='store_true', default=False,
                       help="use express kernel loading (default: false)")
    group.add_argument('--mlator', action='store_true', default=False,
                       help="use hardware to swap output bytes (default: false)")
    mgroup = group.add_mutually_exclusive_group()
    mgroup.add_argument('-f', '--fc-layer', action='store_true', default=False,
                        help="add unload, a fully connected classification layer, and softmax"
                             "in software (default: false; AI85 uses hardware)")
    mgroup.add_argument('--unload', action='store_true', default=False,
                        help="add a 'cnn_unload()' function (default: false)")
    mgroup.add_argument('--softmax', action='store_true', default=False,
                        help="add unload and software softmax functions (default: false)")
    group.add_argument('--boost', metavar='S', default=None,
                       help="dot-separated port and pin that is turned on during CNN run to "
                            "boost the power supply, e.g. --boost 2.5 (default: None)")
    group.add_argument('--energy', action='store_true', default=False,
                       help="insert instrumentation code for energy measurement")

    # File names
    group = parser.add_argument_group('File names')
    group.add_argument('--c-filename', metavar='S',
                       help="C file name base (RTL sim default: 'test' -> 'test.c', "
                            "otherwise 'main' -> 'main.c')")
    group.add_argument('--weight-filename', metavar='S', default='weights.h',
                       help="weight header file name (default: 'weights.h')")
    group.add_argument('--sample-filename', metavar='S', default='sampledata.h',
                       help="sample data header file name (default: 'sampledata.h')")
    group.add_argument('--sample-input', metavar='S', default=None,
                       help="sample data input file name (default: 'tests/sample_dataset.npy')")

    # Streaming and FIFOs
    group = parser.add_argument_group('Streaming and FIFOs')
    group.add_argument('--fifo', action='store_true', default=False,
                       help="use FIFOs to load streaming data (default: false)")
    group.add_argument('--fast-fifo', action='store_true', default=False,
                       help="use fast FIFO to load streaming data"
                            " (implies --fifo; default: false)")
    group.add_argument('--fast-fifo-quad', action='store_true', default=False,
                       help="use fast FIFO in quad fanout mode (implies --fast-fifo; "
                            "default: false)")
    group.add_argument('--slow-load', type=int, metavar='N', default=0,
                       help="slow down FIFO loads (default: 0)")

    # RISC-V
    group = parser.add_argument_group('RISC-V')
    group.add_argument('--riscv', action='store_true', default=False,
                       help="use RISC-V processor (default: false)")
    group.add_argument('--riscv-flash', action='store_true', default=False,
                       help="move kernel/input to Flash (implies --riscv; default: false)")
    group.add_argument('--riscv-cache', action='store_true', default=False,
                       help="enable RISC-V cache (implies --riscv and --riscv-flash; "
                            "default: false)")
    group.add_argument('--riscv-debug', action='store_true', default=False,
                       help="enable RISC-V debug interface (implies --riscv; default: false)")
    group.add_argument('--riscv-disable-debugwait', dest='riscv_debugwait',
                       action='store_false', default=True,
                       help="disable the for loop before calling WFI() (default: use loop)")
    group.add_argument('--riscv-exclusive', action='store_true', default=False,
                       help="exclusive SRAM access for RISC-V (implies --riscv; default: false)")

    # Debug and Logging
    group = parser.add_argument_group('Debug and logging')
    group.add_argument('-v', '--verbose', action='store_true', default=False,
                       help="verbose output (default: false)")
    group.add_argument('-L', '--log', action='store_true', default=False,
                       help="redirect stdout to log file (default: false)")
    group.add_argument('--log-intermediate', action='store_true', default=False,
                       help="log weights/data between layers to .mem files (default: false)")
    group.add_argument('--log-pooling', action='store_true', default=False,
                       help="log unpooled and pooled data between layers in CSV format "
                            "(default: false)")
    group.add_argument('--log-last-only', action='store_false', dest='verbose_all', default=True,
                       help="log data for last layer only (default: all layers)")
    group.add_argument('--log-filename', default='log.txt', metavar='S',
                       help="log file name (default: 'log.txt')")
    group.add_argument('-D', '--debug', action='store_true', default=False,
                       help="debug mode (default: false)")
    group.add_argument('--debug-computation', action='store_true', default=False,
                       help="debug computation -- SLOW (default: false)")
    group.add_argument('--no-error-stop', action='store_true', default=False,
                       help="do not stop on errors (default: stop)")
    group.add_argument('--stop-after', type=int, metavar='N',
                       help="stop after layer")
    group.add_argument('--stop-start', action='store_true', default=False,
                       help="stop and then restart the accelerator (default: false)")
    group.add_argument('--one-shot', action='store_true', default=False,
                       help="use layer-by-layer one-shot mechanism (default: false)")
    group.add_argument('--repeat-layers', type=int, metavar='N', default=1,
                       help="repeat layers N times (default: once)")
    group.add_argument('--clock-trim', metavar='LIST', default=None,
                       help="comma-separated hexadecimal clock trim for IBRO,ISO,IPO; use"
                            "0 to ignore a particular trim")
    group.add_argument('--fixed-input', action='store_true', default=False,
                       help="use fixed 0xaa/0x55 alternating input (default: false)")
    group.add_argument('--max-checklines', type=int, metavar='N', default=None, dest='max_count',
                       help="output only N output check lines (default: all)")
    group.add_argument('--forever', action='store_true', default=False,
                       help="after initial run, repeat CNN forever (default: false)")

    # RTL sim
    group = parser.add_argument_group('RTL simulation')
    group.add_argument('--input-csv', metavar='S',
                       help="input data .csv file name for camera sim")
    group.add_argument('--input-csv-format', type=int, metavar='N', default=888,
                       choices=[555, 565, 888],
                       help="format for .csv input data (555, 565, 888, default: 888)")
    group.add_argument('--input-csv-retrace', type=int, metavar='N', default=camera.RETRACE,
                       help="delay for camera retrace when using .csv input data "
                            f"(default: {camera.RETRACE})")
    group.add_argument('--input-csv-period', metavar='N', default=80,
                       help="period for .csv input data (default: 80)")
    group.add_argument('--input-sync', action='store_true', default=False,
                       help="use synchronous camera input (default: false)")
    group.add_argument('--input-fifo', action='store_true', default=False,
                       help="use software FIFO to buffer input (default: false)")
    group.add_argument('--autogen', default='tests', metavar='S',
                       help="directory location for autogen_list (default: 'tests'); "
                            "don't add if 'None'")
    group.add_argument('--input-filename', default='input', metavar='S',
                       help="input .mem file name base (default: 'input' -> 'input.mem')")
    group.add_argument('--output-filename', default='output', metavar='S',
                       help="output .mem file name base (default: 'output' -> 'output-X.mem')")
    group.add_argument('--runtest-filename', default='run_test.sv', metavar='S',
                       help="run test file name (default: 'run_test.sv')")
    group.add_argument('--legacy-test', action='store_true', default=False,
                       help="enable compatibility for certain old RTL sims (default: false)")
    group.add_argument('--test-dir', metavar='S', required=True,
                       help="set base directory name for auto-filing .mem files")
    group.add_argument('--top-level', default='cnn', metavar='S',
                       help="top level name (default: 'cnn', 'None' for block level)")
    group.add_argument('--queue-name', default='short', metavar='S',
                       help="queue name (default: 'short')")
    group.add_argument('--timeout', type=int, metavar='N',
                       help="set RTL sim timeout (units of 1ms, default based on test)")

    # Streaming
    group = parser.add_argument_group('Streaming tweaks')
    group.add_argument('--overlap-data', '--overwrite-ok', dest='overwrite_ok',
                       action='store_true', default=False,
                       help="allow output to overwrite input (default: warn/stop)")
    group.add_argument('--override-start', type=lambda x: int(x, 0), metavar='N',
                       help="override auto-computed streaming start value (x8 hex)")
    group.add_argument('--increase-start', type=int, default=2, metavar='N',
                       help="add integer to streaming start value (default: 2)")
    group.add_argument('--override-rollover', type=lambda x: int(x, 0), metavar='N',
                       help="override auto-computed streaming rollover value (x8 hex)")
    group.add_argument('--override-delta1', type=lambda x: int(x, 0), metavar='N',
                       help="override auto-computed streaming delta1 value (x8 hex)")
    group.add_argument('--increase-delta1', type=int, default=0, metavar='N',
                       help="add integer to streaming delta1 value (default: 0)")
    group.add_argument('--override-delta2', type=lambda x: int(x, 0), metavar='N',
                       help="override auto-computed streaming delta2 value (x8 hex)")
    group.add_argument('--increase-delta2', type=int, default=0, metavar='N',
                       help="add integer to streaming delta2 value (default: 0)")
    group.add_argument('--ignore-streaming', action='store_true', default=False,
                       help="ignore all 'streaming' layer directives (default: false)")
    group.add_argument('--allow-streaming', action='store_true', default=False,
                       help="allow streaming without use of a FIFO (default: false)")
    group.add_argument('--no-bias', metavar='LIST', default=None,
                       help="comma-separated list of layers where bias values will be ignored "
                            "(default: None)")
    group.add_argument('--streaming-layers', metavar='LIST', default=None,
                       help="comma-separated list of additional streaming layers "
                            "(default: None)")

    # Power
    group = parser.add_argument_group('Power saving')
    group.add_argument('--powerdown', action='store_true', default=False,
                       help="power down unused MRAM instances (default: false)")
    group.add_argument('--deepsleep', action='store_true', default=False,
                       help="put ARM core into deep sleep (default: false)")

    # Hardware settings
    group = parser.add_argument_group('Hardware settings')
    group.add_argument('--max-proc', type=int, metavar='N',
                       help="override maximum number of processors")
    group.add_argument('--input-offset', type=lambda x: int(x, 0), metavar='N',
                       help="input offset (x8 hex, defaults to 0x0000)")
    group.add_argument('--verify-writes', action='store_true', default=False,
                       help="verify write operations (toplevel only, default: false)")
    group.add_argument('--verify-kernels', action='store_true', default=False,
                       help="verify kernels (toplevel only, default: false)")
    group.add_argument('--mlator-noverify', action='store_true', default=False,
                       help="do not check both mlator and non-mlator output (default: false)")
    group.add_argument('--write-zero-registers', action='store_true', default=False,
                       help="write registers even if the value is zero (default: do not write)")
    group.add_argument('--init-tram', action='store_true', default=False,
                       help="initialize TRAM to 0 (default: false)")
    group.add_argument('--zero-sram', action='store_true', default=False,
                       help="zero memories (default: false)")
    group.add_argument('--zero-unused', action='store_true', default=False,
                       help="zero unused registers (default: do not touch)")
    group.add_argument('--apb-base', type=lambda x: int(x, 0), metavar='N',
                       help="APB base address (default: device specific)")
    group.add_argument('--ready-sel', type=int, metavar='N',
                       help="specify memory waitstates")
    group.add_argument('--ready-sel-fifo', type=int, metavar='N',
                       help="specify FIFO waitstates")
    group.add_argument('--ready-sel-aon', type=int, metavar='N',
                       help="specify AON waitstates")

    # Various
    group = parser.add_argument_group('Various')
    group.add_argument('--input-split', type=int, default=1, metavar='N', choices=range(1, 1025),
                       help="split input into N portions (default: don't split)")
    group.add_argument('--synthesize-input', type=int, metavar='N',
                       help="synthesize input data from first 8 lines (default: false)")

    args = parser.parse_args()

    if not args.c_filename:
        args.c_filename = 'main' if args.embedded_code else 'test'

    if not args.device:
        args.device = 84

    if args.no_bias is None:
        args.no_bias = []
    else:
        try:
            args.no_bias = [int(s) for s in args.no_bias.split(',')]
        except ValueError as exc:
            raise ValueError('ERROR: Argument --no-bias must be a comma-separated '
                             'list of integers only') from exc

    if args.clock_trim is not None:
        clock_trim_error = False
        try:
            args.clock_trim = [int(s, 0) for s in args.clock_trim.split(',')]
            if len(args.clock_trim) != 3:
                clock_trim_error = True
        except ValueError:
            clock_trim_error = True
        if clock_trim_error:
            raise ValueError('ERROR: Argument --clock-trim must be a comma-separated '
                             'list of three hexadecimal values (use 0 to ignore a value)')

    if args.boost is not None:
        boost_error = False
        try:
            args.boost = [int(s, 0) for s in args.boost.split('.')]
            if len(args.boost) != 2:
                boost_error = True
        except ValueError:
            boost_error = True
        if boost_error:
            raise ValueError('ERROR: Argument --boost must be a port.pin')

    if args.streaming_layers is not None:
        try:
            args.streaming_layers = [int(s, 0) for s in args.streaming_layers.split(',')]
        except ValueError as exc:
            raise ValueError('ERROR: Argument --streaming-layers must be a comma-separated '
                             'list of integers only') from exc

    if args.top_level == 'None':
        args.top_level = None

    if args.embedded_code is None:
        args.embedded_code = True

    return args
