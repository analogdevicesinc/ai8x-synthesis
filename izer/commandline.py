###################################################################################################
# Copyright (C) 2019-2023 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
###################################################################################################
"""
Command line parser for Tornado CNN
"""
import argparse

from . import camera, state
from .devices import device
from .eprint import wprint
from .tornadocnn import MAX_MAX_LAYERS


def get_parser() -> argparse.Namespace:
    """
    Return an argparse parser.
    """

    parser = argparse.ArgumentParser(description="MAX7800X CNN Generator")

    # Device selection
    group = parser.add_argument_group('Device selection')
    mgroup = group.add_mutually_exclusive_group(required=True)
    mgroup.add_argument('--ai85', action='store_const', const=85, dest='device',
                        help="set device to MAX78000")
    mgroup.add_argument('--device', type=device, metavar='device-name',
                        help="set device")

    # Hardware features
    group = parser.add_argument_group('Hardware features')
    group.add_argument('--avg-pool-rounding', action='store_true', default=False,
                       help="round average pooling results (default: false)")
    group.add_argument('--simple1b', action='store_true', default=False,
                       help="use simple XOR instead of 1-bit multiplication (default: false)")

    # Embedded code
    group = parser.add_argument_group('Embedded code')
    mgroup = group.add_mutually_exclusive_group()
    mgroup.add_argument('-e', '--embedded-code', action='store_true', default=True,
                        help="generate embedded code for device (default)")
    mgroup.add_argument('--rtl', '--rtl-sim', action='store_false', dest='embedded_code',
                        help="generate RTL sim code instead of embedded code (default: false)")
    mgroup.add_argument('--rtl-preload', action='store_true',
                        help="generate RTL sim code with memory preload (default: false)")
    group.add_argument('--rtl-preload-weights', action='store_true',
                       help="generate RTL sim code with weight memory preload (default: false)")
    mgroup = group.add_mutually_exclusive_group()
    mgroup.add_argument('--pipeline', action='store_true', default=None,
                        help="enable pipeline (default: enabled where supported)")
    mgroup.add_argument('--no-pipeline', action='store_false', dest='pipeline',
                        help="disable pipeline")
    mgroup = group.add_mutually_exclusive_group()
    mgroup.add_argument('--pll', action='store_true', default=None,
                        help="enable PLL (default: automatic)")
    mgroup.add_argument('--no-pll', '--apb', action='store_false', dest='pll',
                        help="disable PLL (default: automatic)")
    mgroup = group.add_mutually_exclusive_group()
    mgroup.add_argument('--balance-speed', action='store_true', default=True,
                        help="balance data and weight loading speed and power (default: true)")
    mgroup.add_argument('--max-speed', action='store_false', dest='balance_speed',
                        help="load data and weights as fast as possible (MAX78002 only, "
                             "requires --pll, default: false)")
    mgroup.add_argument('--clock-divider', type=int, metavar='N', choices=[1, 2, 4, 8, 16],
                        help="CNN clock divider (default: 1 or 4, depends on clock source)")
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
    group.add_argument('--debugwait', type=int, default=2, metavar='N',
                       help="set the delay in seconds before calling __WFI() (default: 2)")
    group.add_argument('--define', default='', metavar='S',
                       help="additional #defines for Makefile and auto-generated "
                            "Eclipse project files (default: None; "
                            "use quotes to specify multiple, separated by space)")
    group.add_argument('--define-default-arm', default='MXC_ASSERT_ENABLE ARM_MATH_CM4',
                       metavar='S',
                       help="override default ARM #defines for Makefile and auto-generated "
                            "Eclipse project files (default: 'MXC_ASSERT_ENABLE ARM_MATH_CM4'; "
                            "use quotes to specify multiple, separated by space)")
    group.add_argument('--define-default-riscv', default='MXC_ASSERT_ENABLE RV32', metavar='S',
                       help="override default RISC-V #defines for Makefile and auto-generated "
                            "Eclipse project files (default: 'MXC_ASSERT_ENABLE RV32'; "
                            "use quotes to specify multiple, separated by space)")
    group.add_argument('--eclipse-includes', default='', metavar='S',
                       help="additional includes for auto-generated Eclipse project files "
                            "(default: None)")
    group.add_argument('--eclipse-variables', default='', metavar='S',
                       help="additional variables for auto-generated Eclipse project files "
                            "(default: None)")
    group.add_argument('--eclipse-openocd-args',
                       default='-f interface/cmsis-dap.cfg -f target/##__TARGET_LC__##.cfg',
                       metavar='S',
                       help="set the OpenOCD arguments for auto-generated Eclipse project files "
                            "(default: -f interface/cmsis-dap.cfg -f target/<part>.cfg)")

    # Code generation
    group = parser.add_argument_group('Code generation')
    group.add_argument('--overwrite', action='store_true', default=False,
                       help="overwrite destination if it exists (default: abort)")
    mgroup = group.add_mutually_exclusive_group()
    mgroup.add_argument('--compact-data', action='store_true', default=True,
                        help="use memcpy() to load input data in order to save code space "
                             "(default)")
    mgroup.add_argument('--no-compact-data', action='store_false', dest='compact_data',
                        help="inline input data loader (default: false)")
    group.add_argument('--compact-weights', action='store_true', default=False,
                       help="use memcpy() to load weights in order to save code space")
    mgroup = group.add_mutually_exclusive_group()
    mgroup.add_argument('--mexpress', action='store_true', default=None,
                        help="use express kernel loading (default: true)")
    mgroup.add_argument('--no-mexpress', action='store_false', dest='mexpress',
                        help="disable express kernel loading")
    group.add_argument('--mlator', action='store_true', default=False,
                       help="use hardware to swap output bytes (default: false)")
    group.add_argument('--unroll-mlator', type=int, metavar='N', default=8,
                       help="number of assignments per loop iteration for mlator output "
                            "(default: 8)")
    group.add_argument('--unroll-8bit', type=int, metavar='N', default=1,
                       help="number of assignments per loop iteration for 8-bit output "
                            "(default: 1)")
    group.add_argument('--unroll-wide', type=int, metavar='N', default=8,
                       help="number of assignments per loop iteration for wide output "
                            "(default: 8)")
    group.add_argument('--softmax', action='store_true', default=False,
                       help="add software softmax function (default: false)")
    mgroup = group.add_mutually_exclusive_group()
    mgroup.add_argument('--unload', action='store_true', default=None,
                        help="enable use of cnn_unload() function (default)")
    mgroup.add_argument('--no-unload', dest='unload', action='store_false',
                        help="disable use of cnn_unload() function (default: enabled)")
    group.add_argument('--no-kat', dest='generate_kat', action='store_false', default=True,
                       help="disable known-answer test generation (KAT) (default: enabled)")
    group.add_argument('--boost', metavar='S', default=None,
                       help="dot-separated port and pin that is turned on during CNN run to "
                            "boost the power supply, e.g. --boost 2.5 (default: None)")
    group.add_argument('--start-layer', type=int, metavar='N', default=0,
                       help="set starting layer (default: 0)")
    group.add_argument('--no-wfi', dest='wfi', action='store_false', default=True,
                       help="do not use _WFI() (default: _WFI() is used)")
    group.add_argument('--timer', type=int, metavar='N',
                       help="use timer to time the inference (default: off, supply timer number)")
    group.add_argument('--no-timer', action='store_true', default=False,
                       help="ignore --timer argument(s)")
    group.add_argument('--energy', action='store_true', default=False,
                       help="insert instrumentation code for energy measurement")
    group.add_argument('--switch-delay', dest='enable_delay', type=int, metavar='N', default=None,
                       help="set delay in msec after cnn_enable() for load switches (default: 0"
                            " on MAX78000, 10 on MAX78002)")
    group.add_argument('--output-width', type=int, default=None,
                       choices=[8, 32],
                       help="override `output_width` for the final layer (default: use YAML)")
    group.add_argument('--no-deduplicate-weights', action='store_true', default=False,
                       help="do not reuse weights (default: enabled)")

    # File names
    group = parser.add_argument_group('File names')
    group.add_argument('--c-filename', metavar='S',
                       help="C file name base (RTL sim default: 'test' -> 'test.c', "
                            "otherwise 'main' -> 'main.c')")
    group.add_argument('--api-filename', metavar='S', default='cnn.c',
                       help="C library file name (default: 'cnn.c', 'none' to disable)")
    group.add_argument('--weight-filename', metavar='S', default='weights.h',
                       help="weight header file name (default: 'weights.h')")
    group.add_argument('--sample-filename', metavar='S', default='sampledata.h',
                       help="sample data header file name (default: 'sampledata.h')")
    group.add_argument('--sample-input', metavar='S', default=None,
                       help="sample data input file name (default: 'tests/sample_dataset.npy')")
    group.add_argument('--sample-output-filename', dest='result_filename', metavar='S',
                       default=None,
                       help="sample result header file name (default: 'sampleoutput.h', use "
                            "'None' to inline code)")
    group.add_argument('--sample-numpy-filename', dest='result_numpy', metavar='S',
                       help="save sample result as NumPy file (default: disabled)")

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
    group.add_argument('--no-fifo-wait', dest='fifo_wait', action='store_false', default=True,
                       help="do not check the FIFO for available space (requires matching source "
                            "speed to inference, default: false)")
    group.add_argument('--fifo-go', action='store_true', default=False,
                       help="start processing before first FIFO push (default: false)")
    group.add_argument('--slow-load', type=int, metavar='N', default=0,
                       help="slow down FIFO loads (default: 0)")
    group.add_argument('--debug-new-streaming', action='store_true', default=True,
                       help="modify streaming equation (default: false)")

    # RISC-V
    group = parser.add_argument_group('RISC-V')
    group.add_argument('--riscv', action='store_true', default=False,
                       help="use RISC-V processor (default: false)")
    mgroup = group.add_mutually_exclusive_group()
    mgroup.add_argument('--riscv-flash', action='store_true', default=None,
                        help="move kernel/input to Flash (implies --riscv; default: true)")
    mgroup.add_argument('--no-riscv-flash', action='store_false', dest='riscv_flash',
                        help="disable --riscv-flash")
    mgroup = group.add_mutually_exclusive_group()
    mgroup.add_argument('--riscv-cache', action='store_true', default=None,
                        help="enable RISC-V cache (implies --riscv and --riscv-flash; "
                             "default: true)")
    mgroup.add_argument('--no-riscv-cache', action='store_false', dest='riscv_cache',
                        help="disable RISC-V cache")
    group.add_argument('--riscv-debug', action='store_true', default=False,
                       help="enable RISC-V debug interface (implies --riscv; default: false)")
    group.add_argument('--riscv-exclusive', action='store_true', default=False,
                       help="exclusive SRAM access for RISC-V (implies --riscv; default: false)")

    # Debug and Logging
    group = parser.add_argument_group('Debug and logging')
    group.add_argument('-v', '--verbose', action='store_true', default=False,
                       help="verbose output (default: false)")
    group.add_argument('-L', '--log', action='store_true', default=None,
                       help="redirect stdout to log file (default)")
    group.add_argument('--no-log', dest='log', action='store_false',
                       help="do not redirect stdout to log file (default: false)")
    group.add_argument('--no-progress', dest='display_progress', action='store_false',
                       default=True, help="do not display progress bars (default: show)")
    group.add_argument('--log-intermediate', action='store_true', default=False,
                       help="log weights/data between layers to .mem files (default: false)")
    group.add_argument('--log-pooling', action='store_true', default=False,
                       help="log unpooled and pooled data between layers in CSV format "
                            "(default: false)")
    group.add_argument('--short-log', '--log-last-only', '--verbose-all',
                       action='store_false', dest='verbose_all', default=True,
                       help="log data for output layers only (default: all layers)")
    group.add_argument('--log-filename', default='log.txt', metavar='S',
                       help="log file name (default: 'log.txt')")
    group.add_argument('-D', '--debug', action='store_true', default=False,
                       help="debug mode (default: false)")
    group.add_argument('--debug-computation', action='store_true', default=False,
                       help="debug computation -- SLOW (default: false)")
    group.add_argument('--debug-latency', action='store_true', default=False,
                       help="debug latency calculations (default: false)")
    group.add_argument('--no-error-stop', action='store_true', default=False,
                       help="do not stop on errors (default: stop)")
    group.add_argument('--stop-after', type=int, metavar='N',
                       help="stop after layer")
    group.add_argument('--skip-checkpoint-layers', type=int, metavar='N', default=0,
                       help="ignore first N layers in the checkpoint (default: 0)")
    group.add_argument('--skip-yaml-layers', type=int, metavar='N', default=0,
                       help="ignore first N layers in the yaml file (default: 0)")
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
    group.add_argument('--reshape-inputs', action='store_true', default=False,
                       help="drop data channel dimensions to match weights (default: false)")
    group.add_argument('--forever', action='store_true', default=False,
                       help="after initial run, repeat CNN forever (default: false)")
    group.add_argument('--link-layer', action='store_true', default=False,
                       help="always use the link layer feature (default: false)")
    group.add_argument('--read-ahead', dest='rd_ahead', action='store_true', default=False,
                       help="set the rd_ahead bit (default: false)")
    group.add_argument('--calcx4', action='store_true', default=False,
                       help="rearrange kernels and set the calcx4 bit (default: false)")
    group.add_argument('--ext-rdy', action='store_true', default=False,
                       help="set ext_rdy bit (default: false)")
    group.add_argument('--weight-start', type=int, metavar='N', default=0,
                       help="specify start offset for weights (debug, default: 0)")
    group.add_argument('--ignore-bias-groups', action='store_true', default=False,
                       help="do not force `bias_group` to use an active group (default: false)")
    group.add_argument('--kernel-format', default='{0:4}', metavar='S',
                       help="print format for kernels (default: '{0:4}')")
    group.add_argument('--debug-snoop', action='store_true', default=False,
                       help="insert snoop register debug code (default: False)")
    group.add_argument('--snoop-loop', action='store_true', default=False,
                       help="insert snoop loop (default: false)")
    group.add_argument('--ignore-hw-limits', action='store_true', default=False,
                       help="ignore certain hardware limits (default: false)")
    group.add_argument('--ignore-bn', action='store_true', default=False,
                       help="ignore BatchNorm weights in checkpoint file (default: false)")
    group.add_argument('--ignore-activation', action='store_true', default=False,
                       help="ignore activations in YAML file (default: false)")
    group.add_argument('--ignore-energy-warning', action='store_true', default=False,
                       help="do not show energy and performance hints (default: show)")
    group.add_argument('--ignore-mlator-warning', action='store_true', default=False,
                       help="do not show mlator hints (default: show)")
    # group.add_argument('--no-greedy-kernel', action='store_false',
    #                    dest='greedy_kernel_allocator', default=True,
    #                    help="do not use greedy kernel memory allocator (default: use)")
    mgroup = group.add_mutually_exclusive_group()
    mgroup.add_argument('--new-kernel-loader', action='store_true', default=True,
                        help="use new kernel loader (default)")
    mgroup.add_argument('--old-kernel-loader', dest='new_kernel_loader', action='store_false',
                        default=None, help="use alternate old kernel loader")
    group.add_argument('--weight-input', metavar='S', default=None,
                       help="weight input file name (development only, "
                            "default: 'tests/weight_test_dataset.npy')")
    group.add_argument('--bias-input', metavar='S', default=None,
                       help="bias input file name (development only, "
                            "default: 'tests/bias_dataset.npy')")

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
    group.add_argument('--input-pix-clk', metavar='N', default=9,
                       help="pixel clock for .csv input data (default: 9)")
    group.add_argument('--input-sync', action='store_true', default=False,
                       help="use synchronous camera input (default: false)")
    group.add_argument('--input-fifo', action='store_true', default=False,
                       help="use software FIFO to buffer input (default: false)")
    group.add_argument('--autogen', default='None', metavar='S',
                       help="directory location for autogen_list (default: None)")
    group.add_argument('--autogen_list', default='autogen_list', metavar='S',
                       help="file name for autogen_list")
    group.add_argument('--input-filename', default='input', metavar='S',
                       help="input .mem file name base (default: 'input' -> 'input.mem')")
    group.add_argument('--output-filename', default='output', metavar='S',
                       help="output .mem or .csv file name base (default: 'output' -> "
                            "'output-X.mem' or 'output.csv')")
    group.add_argument('--output-config-filename', default='config', metavar='S',
                       help="output config file name base (default: 'config' -> 'config.csv')")
    group.add_argument('--output-data-filename', default='data', metavar='S',
                       help="output data file name base (default: 'data' -> 'data.npy')")
    group.add_argument('--output-weights-filename', default='weights', metavar='S',
                       help="output weights file name base (default: 'weights' -> 'weights.npy')")
    group.add_argument('--output-bias-filename', default='bias', metavar='S',
                       help="output bias file name base (default: 'bias' -> 'bias.npy')")
    group.add_argument('--output-pass-filename', default=None, metavar='S',
                       help="output pass data file name base (default: None)")
    group.add_argument('--runtest-filename', default='run_test.sv', metavar='S',
                       help="run test file name (default: 'run_test.sv')")
    group.add_argument('--legacy-test', action='store_true', default=False,
                       help="enable compatibility for certain old RTL sims (default: false)")
    group.add_argument('--legacy-kernels', action='store_true', default=False,
                       help="use old, less efficient kernel allocation for certain old RTL sims"
                            " (default: false)")
    group.add_argument('--test-dir', metavar='S', required=True,
                       help="set base directory name for auto-filing .mem files")
    group.add_argument('--top-level', default='cnn', metavar='S',
                       help="top level name (default: 'cnn', 'None' for block level)")
    group.add_argument('--queue-name', default=None, metavar='S',
                       help="queue name (default: 'short')")
    group.add_argument('--timeout', type=int, metavar='N',
                       help="set RTL sim timeout (units of 1ms, default based on test)")
    group.add_argument('--result-output', action='store_true', default=False,
                       help="write expected output to memory dumps instead of inline code"
                            " (default: false)")

    # Streaming
    group = parser.add_argument_group('Streaming tweaks')
    group.add_argument('--overlap-data',
                       dest='overwrite_ok', action='store_true', default=False,
                       help="allow output to overwrite input (default: warn/stop)")
    group.add_argument('--override-start', type=lambda x: int(x, 0), metavar='N',
                       help="override auto-computed streaming start value (x8 hex)")
    group.add_argument('--increase-start', type=lambda x: int(x, 0), default=2, metavar='N',
                       help="add integer to streaming start value (default: 2)")
    group.add_argument('--override-rollover', type=lambda x: int(x, 0), metavar='N',
                       help="override auto-computed streaming rollover value (x8 hex)")
    group.add_argument('--override-delta1', type=lambda x: int(x, 0), metavar='N',
                       help="override auto-computed streaming delta1 value (x8 hex)")
    group.add_argument('--increase-delta1', type=lambda x: int(x, 0), default=0, metavar='N',
                       help="add integer to streaming delta1 value (default: 0)")
    group.add_argument('--override-delta2', type=lambda x: int(x, 0), metavar='N',
                       help="override auto-computed streaming delta2 value (x8 hex)")
    group.add_argument('--increase-delta2', type=lambda x: int(x, 0), default=0, metavar='N',
                       help="add integer to streaming delta2 value (default: 0)")
    group.add_argument('--ignore-streaming', action='store_true', default=False,
                       help="ignore all 'streaming' layer directives (default: false)")
    group.add_argument('--allow-streaming', action='store_true', default=False,
                       help="allow streaming without use of a FIFO (default: false)")
    group.add_argument('--no-bias', metavar='[LIST]', nargs='?',
                       const=list(range(MAX_MAX_LAYERS)),
                       help="comma-separated list of layers where bias values will be ignored, "
                            "or no argument for all layers (default: None)")
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
    group.add_argument('--pretend-zero-sram', action='store_true', default=False,
                       help="simulate --zero-sram, but block BIST (default: false)")
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
    group.add_argument('--synthesize-input', type=lambda x: int(x, 0), metavar='N',
                       help="synthesize input data from first `--synthesize-words` words and add "
                            "N to each subsequent set of `--synthesize-words` 32-bit words "
                            "(default: false)")
    group.add_argument('--synthesize-words', type=int, metavar='N', default=8,
                       help="number of input words to use (default all or 8)")
    group.add_argument('--max-verify-length', '--max-checklines',
                       type=int, metavar='N', default=None, dest='max_count',
                       help="output only N output check lines (default: all)")
    group.add_argument('--no-version-check', action='store_true', default=False,
                       help='do not check GitHub for newer versions of the repository')
    group.add_argument('--version-check-interval', type=int, metavar='HOURS', default=24,
                       help='version check update interval (hours), default = 24')
    group.add_argument('--upstream', metavar='REPO', default="MaximIntegratedAI/ai8x-synthesis",
                       help='GitHub repository name for update checking')

    args = parser.parse_args()

    if args.rtl_preload:
        args.embedded_code = False
    if args.verify_kernels or (args.verify_writes and args.new_kernel_loader) \
       or args.mexpress is not None:
        args.rtl_preload_weights = False
    if args.rtl_preload_weights:
        args.mexpress = False
    if args.embedded_code is None:
        args.embedded_code = True
    if not args.embedded_code:
        args.softmax = False
        args.energy = False
    if args.mexpress is None:
        args.mexpress = True
    if args.mlator:
        args.result_output = False

    if not args.c_filename:
        args.c_filename = 'main' if args.embedded_code else 'test'

    # Set default
    if args.log is None:
        args.log = True
    if args.unload is None:
        args.unload = True

    if args.no_bias is None:
        args.no_bias = []
    else:
        try:
            if isinstance(args.no_bias, list):
                args.no_bias = [int(s) for s in args.no_bias]
            else:
                args.no_bias = [int(s) for s in args.no_bias.split(',')]
        except ValueError as exc:
            raise ValueError('ERROR: Argument `--no-bias` must be a comma-separated '
                             'list of integers only, or no argument') from exc

    if args.clock_trim is not None:
        clock_trim_error = False
        try:
            args.clock_trim = [int(s, 0) for s in args.clock_trim.split(',')]
            if len(args.clock_trim) != 3:
                clock_trim_error = True
        except ValueError:
            clock_trim_error = True
        if clock_trim_error:
            raise ValueError('ERROR: Argument `--clock-trim` must be a comma-separated '
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
            raise ValueError('ERROR: Argument `--boost` must be a port.pin')

    if args.streaming_layers is not None:
        try:
            args.streaming_layers = [int(s, 0) for s in args.streaming_layers.split(',')]
        except ValueError as exc:
            raise ValueError('ERROR: Argument `--streaming-layers` must be a comma-separated '
                             'list of integers only') from exc

    if args.top_level == 'None':
        args.top_level = None

    if args.riscv_flash is None:
        args.riscv_flash = args.riscv
    if args.riscv_cache is None:
        args.riscv_cache = args.riscv

    if args.allow_streaming:
        wprint('`--allow-streaming` is unsupported.')

    if args.result_filename is None:
        args.result_filename = 'sampleoutput.h' if args.embedded_code else None
    elif args.result_filename.lower() == 'none':
        args.result_filename = None

    if args.define != '':
        args.define = "-D" + " -D".join(args.define.split(' '))

    if args.define_default_arm != '':
        args.define_default_arm = "-D" + " -D".join(args.define_default_arm.split(' '))

    if args.define_default_riscv != '':
        args.define_default_riscv = "-D" + " -D".join(args.define_default_riscv.split(' '))

    if args.no_timer:
        args.timer = None

    if args.timer is not None and args.energy:
        wprint('`--timer` is ignored when using `--energy`. Remove the `--timer` argument or '
               'add `--no-timer` to suppress this message.')
        args.timer = None

    return args


def set_state(args: argparse.Namespace) -> None:
    """
    Set configuration state based on command line arguments.

    :param args: list of command line arguments
    """

    state.allow_streaming = args.allow_streaming
    if args.apb_base:
        state.apb_base = args.apb_base
    state.api_filename = args.api_filename
    state.avg_pool_rounding = args.avg_pool_rounding
    state.balance_power = args.balance_speed
    state.base_directory = args.test_dir
    state.block_mode = not args.top_level
    state.board_name = args.board_name
    state.boost = args.boost
    state.c_filename = args.c_filename
    state.calcx4 = args.calcx4
    state.clock_divider = args.clock_divider
    state.clock_trim = args.clock_trim
    state.compact_data = args.compact_data and \
        (not args.rtl_preload or args.fifo or args.fast_fifo or args.fast_fifo_quad)
    state.compact_weights = args.compact_weights
    state.debug = args.debug
    state.debug_computation = args.debug_computation
    state.debug_latency = args.debug_latency
    state.debug_new_streaming = args.debug_new_streaming
    state.debug_snoop = args.debug_snoop
    state.debug_wait = args.debugwait
    state.defines = args.define
    state.defines_arm = args.define_default_arm
    state.defines_riscv = args.define_default_riscv
    state.display_progress = args.display_progress
    state.eclipse_includes = args.eclipse_includes
    state.eclipse_openocd_args = args.eclipse_openocd_args
    state.eclipse_variables = args.eclipse_variables
    state.embedded_code = args.embedded_code
    state.enable_delay = args.enable_delay
    state.energy_warning = not args.ignore_energy_warning
    state.ext_rdy = args.ext_rdy
    state.fast_fifo = args.fast_fifo
    state.fast_fifo_quad = args.fast_fifo_quad
    state.fifo = args.fifo
    state.fifo_go = args.fifo_go
    state.fifo_wait = args.fifo_wait
    state.fixed_input = args.fixed_input
    state.forever = args.forever and args.embedded_code
    state.generate_kat = args.generate_kat
    # state.greedy_kernel_allocator = args.greedy_kernel_allocator
    state.ignore_activation = args.ignore_activation
    state.ignore_bias_groups = args.ignore_bias_groups
    state.ignore_bn = args.ignore_bn
    state.ignore_hw_limits = args.ignore_hw_limits
    state.increase_delta1 = args.increase_delta1
    state.increase_delta2 = args.increase_delta2
    state.increase_start = args.increase_start
    state.init_tram = args.init_tram
    state.input_csv = args.input_csv
    state.input_csv_format = args.input_csv_format
    state.input_csv_period = args.input_csv_period
    state.input_csv_retrace = args.input_csv_retrace
    state.input_fifo = args.input_fifo
    state.input_filename = args.input_filename
    state.input_pix_clk = args.input_pix_clk
    state.input_sync = args.input_sync
    state.kernel_format = args.kernel_format
    state.legacy_kernels = args.legacy_kernels
    state.legacy_test = args.legacy_test
    state.link_layer = args.link_layer
    state.log = args.log
    state.log_filename = args.log_filename
    state.log_intermediate = args.log_intermediate
    state.log_pooling = args.log_pooling
    state.max_count = args.max_count
    state.measure_energy = args.energy
    state.mexpress = args.mexpress
    state.mlator = args.mlator
    state.mlator_chunk = args.unroll_mlator
    state.mlator_noverify = args.mlator_noverify
    state.mlator_warning = not args.ignore_mlator_warning
    state.narrow_chunk = args.unroll_8bit
    state.new_kernel_loader = args.new_kernel_loader
    state.deduplicate_weights = not args.no_deduplicate_weights
    state.no_error_stop = args.no_error_stop
    state.oneshot = args.one_shot
    state.output_filename = args.output_filename
    state.output_config_filename = args.output_config_filename
    state.output_data_filename = args.output_data_filename
    state.output_weights_filename = args.output_weights_filename
    state.output_bias_filename = args.output_bias_filename
    state.output_pass_filename = args.output_pass_filename
    state.override_delta1 = args.override_delta1
    state.override_delta2 = args.override_delta2
    state.override_rollover = args.override_rollover
    state.override_start = args.override_start
    state.overwrite = args.overwrite
    state.overwrite_ok = args.overwrite_ok
    state.pipeline = args.pipeline
    state.pll = args.pll
    state.powerdown = args.powerdown
    state.prefix = args.prefix
    state.pretend_zero_sram = args.pretend_zero_sram
    state.repeat_layers = args.repeat_layers
    state.reshape_inputs = args.reshape_inputs
    state.result_filename = args.result_filename
    state.result_numpy = args.result_numpy
    state.result_output = args.result_output
    state.riscv = args.riscv
    state.riscv_cache = args.riscv_cache
    state.riscv_debug = args.riscv_debug
    state.riscv_exclusive = args.riscv_exclusive
    state.riscv_flash = args.riscv_flash
    state.rtl_preload = args.rtl_preload
    state.rtl_preload_weights = args.rtl_preload_weights
    state.runtest_filename = args.runtest_filename
    state.sample_filename = args.sample_filename
    state.simple1b = args.simple1b
    state.sleep = args.deepsleep
    state.slow_load = args.slow_load
    state.snoop_loop = args.snoop_loop
    state.softmax = args.softmax
    state.split = args.input_split
    state.start_layer = args.start_layer
    state.stopstart = args.stop_start
    state.synthesize_input = args.synthesize_input
    state.synthesize_words = args.synthesize_words
    state.test_dir = args.test_dir
    state.timeout = args.timeout
    state.timer = args.timer
    state.unload = args.unload
    state.verbose = args.verbose
    state.verbose_all = args.verbose_all
    state.verify_kernels = args.verify_kernels or args.verify_writes and args.new_kernel_loader
    state.verify_writes = args.verify_writes
    state.weight_filename = args.weight_filename
    state.weight_start = args.weight_start
    state.wfi = args.wfi
    state.wide_chunk = args.unroll_wide
    state.write_zero_regs = args.write_zero_registers
    state.zero_sram = args.zero_sram
    state.zero_unused = args.zero_unused
