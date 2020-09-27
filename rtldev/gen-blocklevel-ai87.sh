#!/bin/sh
DEVICE="--device 78002"
TARGET="blocklevel-ai87"
PREFIX="ai87"
SHORT_LOG="--log-last-only"

./ai8xize.py --verbose --autogen $TARGET -L --test-dir $TARGET --prefix $PREFIX-cifar --checkpoint-file trained/ai84-cifar10.pth.tar --config-file networks/cifar10-hwc.yaml $DEVICE
./ai8xize.py --verbose --autogen $TARGET -L --test-dir $TARGET --prefix $PREFIX-singlebyte-hwc --config-file tests/test-singlebyte-hwc.yaml $DEVICE
./ai8xize.py --verbose --autogen $TARGET -L --test-dir $TARGET --prefix $PREFIX-kmax_bmax_dmax --config-file tests/test-max.yaml $DEVICE
./ai8xize.py --verbose --autogen $TARGET -L --test-dir $TARGET --prefix $PREFIX-passthru-2-pool --config-file tests/test-passthrough-2-pool.yaml $DEVICE
./ai8xize.py --verbose --autogen $TARGET -L --test-dir $TARGET --prefix $PREFIX-conv1d-multilayer-q1248 --config-file tests/test-conv1d-multilayer-q1248.yaml $DEVICE --stop-after 2
./ai8xize.py --verbose --autogen $TARGET -L --test-dir $TARGET --prefix $PREFIX-eltwise-pool --config-file tests/test-eltwiseadd-pool.yaml $DEVICE --legacy-test
./ai8xize.py --verbose --autogen $TARGET -L --test-dir $TARGET --prefix $PREFIX-eltwise-poolafter-avg --config-file tests/test-eltwiseadd-poolafter-avg.yaml $DEVICE --legacy-test
# ./ai8xize.py --verbose --autogen $TARGET -L --test-dir $TARGET --prefix $PREFIX-fifostream-eltwise --config-file tests/test-fifostream-eltwise.yaml $DEVICE --fifo --increase-start 1
./ai8xize.py --verbose --autogen $TARGET -L --test-dir $TARGET --prefix $PREFIX-passthroughmp --config-file tests/test-passthroughmultipass.yaml $DEVICE
./ai8xize.py --verbose --autogen $TARGET -L --test-dir $TARGET --prefix $PREFIX-wide3to508to3 --config-file tests/test-wide3to508to3.yaml $DEVICE
./ai8xize.py --verbose --autogen $TARGET -L --test-dir $TARGET --prefix $PREFIX-mlpflatten12to2 --config-file tests/test-mlpflatten12to2.yaml $DEVICE --debug-computation --debug
./ai8xize.py --verbose --autogen $TARGET -L --test-dir $TARGET --prefix $PREFIX-conv1d-2 --config-file tests/test-conv1d-2.yaml $DEVICE
./ai8xize.py --verbose --autogen $TARGET -L --test-dir $TARGET --prefix $PREFIX-conv1x1 --config-file tests/test-conv1x1.yaml $DEVICE
./ai8xize.py --verbose --autogen $TARGET -L --test-dir $TARGET --prefix $PREFIX-upscale --config-file tests/test-upscale.yaml $DEVICE --stop-after 0
# ./ai8xize.py --verbose --autogen $TARGET -L --test-dir $TARGET --prefix $PREFIX-nonsquare-mexpress-mlator --config-file tests/test-nonsquare.yaml $DEVICE --mexpress --mlator
# ./ai8xize.py --verbose --autogen $TARGET -L --test-dir $TARGET --prefix $PREFIX-riscv-qfastfifostream-32-hwc --config-file tests/test-fifostream-32-hwc.yaml $DEVICE --fifo --riscv --riscv-flash --fast-fifo-quad --riscv-cache --input-fifo
