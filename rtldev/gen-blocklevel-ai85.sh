#!/bin/sh
DEVICE="78000"
TARGET="blocklevel-ai85"
PREFIX="ai85"

./ai8xize.py --verbose --autogen $TARGET --log --top-level None --test-dir $TARGET --prefix $PREFIX-cifar --checkpoint-file trained/ai84-cifar10.pth.tar --config-file networks/cifar10-hwc.yaml --device "$DEVICE"
./ai8xize.py --verbose --autogen $TARGET --log --top-level None --test-dir $TARGET --prefix $PREFIX-singlebyte-hwc --config-file tests/test-singlebyte-hwc.yaml --device "$DEVICE"
./ai8xize.py --verbose --autogen $TARGET --log --top-level None --test-dir $TARGET --prefix $PREFIX-kmax_bmax_dmax --config-file tests/test-max.yaml --device "$DEVICE"
./ai8xize.py --verbose --autogen $TARGET --log --top-level None --test-dir $TARGET --prefix $PREFIX-passthru-2-pool --config-file tests/test-passthrough-2-pool.yaml --device "$DEVICE"
./ai8xize.py --verbose --autogen $TARGET --log --top-level None --test-dir $TARGET --prefix $PREFIX-conv1d-multilayer-q1248 --config-file tests/test-conv1d-multilayer-q1248.yaml --device "$DEVICE" --stop-after 2
./ai8xize.py --verbose --autogen $TARGET --log --top-level None --test-dir $TARGET --prefix $PREFIX-eltwise-pool --config-file tests/test-eltwiseadd-pool.yaml --device "$DEVICE" --legacy-test
./ai8xize.py --verbose --autogen $TARGET --log --top-level None --test-dir $TARGET --prefix $PREFIX-eltwise-poolafter-avg --config-file tests/test-eltwiseadd-poolafter-avg.yaml --device "$DEVICE" --legacy-test
# ./ai8xize.py --verbose --autogen $TARGET --log --top-level None --test-dir $TARGET --prefix $PREFIX-fifostream-eltwise --config-file tests/test-fifostream-eltwise.yaml --device "$DEVICE" --fifo --increase-start 1
./ai8xize.py --verbose --autogen $TARGET --log --top-level None --test-dir $TARGET --prefix $PREFIX-passthroughmp --config-file tests/test-passthroughmultipass.yaml --device "$DEVICE"
./ai8xize.py --verbose --autogen $TARGET --log --top-level None --test-dir $TARGET --prefix $PREFIX-wide3to508to3 --config-file tests/test-wide3to508to3.yaml --device "$DEVICE"
./ai8xize.py --verbose --autogen $TARGET --log --top-level None --test-dir $TARGET --prefix $PREFIX-mlpflatten12to2 --config-file tests/test-mlpflatten12to2.yaml --device "$DEVICE" --debug-computation --debug
./ai8xize.py --verbose --autogen $TARGET --log --top-level None --test-dir $TARGET --prefix $PREFIX-conv1d-2 --config-file tests/test-conv1d-2.yaml --device "$DEVICE"
./ai8xize.py --verbose --autogen $TARGET --log --top-level None --test-dir $TARGET --prefix $PREFIX-conv1x1 --config-file tests/test-conv1x1.yaml --device "$DEVICE"
./ai8xize.py --verbose --autogen $TARGET --log --top-level None --test-dir $TARGET --prefix $PREFIX-upscale --config-file tests/test-upscale.yaml --device "$DEVICE" --stop-after 0
# ./ai8xize.py --verbose --autogen $TARGET --log --top-level None --test-dir $TARGET --prefix $PREFIX-nonsquare-mexpress-mlator --config-file tests/test-nonsquare.yaml --device "$DEVICE" --mexpress --mlator
# ./ai8xize.py --verbose --autogen $TARGET --log --top-level None --test-dir $TARGET --prefix $PREFIX-riscv-qfastfifostream-32-hwc --config-file tests/test-fifostream-32-hwc.yaml --device "$DEVICE" --fifo --riscv --riscv-flash --fast-fifo-quad --riscv-cache --input-fifo
