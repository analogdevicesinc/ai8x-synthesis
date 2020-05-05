#!/bin/sh
CLOCK_TRIM=""
# Example: CLOCK_TRIM="--clock-trim 0x173,0x5e,0x14"
SDK_TARGET="sdk/Examples/MAX78000/CNN"

# Update common code and header files
cp device-ai85/softmax.c $SDK_TARGET/Common/
cp device-ai85/*.h $SDK_TARGET/Common/

./ai8xize.py -e --verbose --top-level cnn -L --test-dir $SDK_TARGET --prefix mnist --checkpoint-file trained/ai85-mnist.pth.tar --config-file networks/mnist-chw-ai85.yaml --device 85 --compact-data --mexpress --softmax --display-checkpoint $CLOCK_TRIM
./ai8xize.py -e --verbose --top-level cnn -L --test-dir $SDK_TARGET --prefix mnist-riscv --checkpoint-file trained/ai85-mnist.pth.tar --config-file networks/mnist-chw-ai85.yaml --device 85 --compact-data --mexpress --softmax --display-checkpoint $CLOCK_TRIM --riscv --riscv-flash --riscv-cache --riscv-debug
./ai8xize.py -e --verbose --top-level cnn -L --test-dir $SDK_TARGET --prefix cifar-10 --checkpoint-file trained/ai85-cifar10-bias.pth.tar --config-file networks/cifar10-hwc.yaml --device 85 --verify-writes --compact-data --mexpress $CLOCK_TRIM

# ./ai8xize.py -e --verbose --top-level cnn -L --test-dir demos --prefix ai85-fifostream-cifar2-hwc --checkpoint-file trained/ai84-cifar10.pth.tar --config-file tests/test-fifostream-cifar-hwc.yaml --device 85 --fifo $CLOCK_TRIM
