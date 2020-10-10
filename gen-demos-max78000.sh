#!/bin/sh
DEVICE="MAX78000"
SDK_TARGET="sdk/Examples/$DEVICE/CNN"
EXTRA_ARGS=""
# Example: EXTRA_ARGS="--boost 2.5"

# Update common code and header files
mkdir -p $SDK_TARGET/Common/
cp assets/device-ai85/softmax.c $SDK_TARGET/Common/
cp assets/device-ai85/*.h $SDK_TARGET/Common/

./ai8xize.py -e --verbose --top-level cnn -L --test-dir $SDK_TARGET --prefix mnist --checkpoint-file trained/ai85-mnist.pth.tar --config-file networks/mnist-chw-ai85.yaml --device "$DEVICE" --compact-data --mexpress --softmax --display-checkpoint $EXTRA_ARGS "$@"
./ai8xize.py -e --verbose --top-level cnn -L --test-dir $SDK_TARGET --prefix mnist-riscv --checkpoint-file trained/ai85-mnist.pth.tar --config-file networks/mnist-chw-ai85.yaml --device "$DEVICE" --compact-data --mexpress --softmax --display-checkpoint $EXTRA_ARGS --riscv --riscv-flash --riscv-cache --riscv-debug "$@"
./ai8xize.py -e --verbose --top-level cnn -L --test-dir $SDK_TARGET --prefix cifar-10 --checkpoint-file trained/ai85-cifar10.pth.tar --config-file networks/cifar10-hwc-ai85.yaml --device "$DEVICE" --compact-data --mexpress --display-checkpoint $EXTRA_ARGS "$@"
./ai8xize.py -e --verbose --top-level cnn -L --test-dir $SDK_TARGET --prefix cifar-100 --checkpoint-file trained/ai85-cifar100.pth.tar --config-file networks/cifar100-simple.yaml --device "$DEVICE" --compact-data --mexpress --softmax --display-checkpoint $EXTRA_ARGS --boost 2.5 "$@"
./ai8xize.py -e --verbose --top-level cnn -L --test-dir $SDK_TARGET --prefix kws20 --checkpoint-file trained/ai85-kws20.pth.tar --config-file networks/kws20-hwc.yaml --device "$DEVICE" --compact-data --mexpress --softmax --embedded-code --display-checkpoint $EXTRA_ARGS "$@"
./ai8xize.py -e --verbose --top-level cnn -L --test-dir $SDK_TARGET --prefix faceid --checkpoint-file trained/ai85-faceid.pth.tar --config-file networks/faceid.yaml --device "$DEVICE" --fifo --compact-data --mexpress --display-checkpoint --unload $EXTRA_ARGS "$@"
./ai8xize.py -e --verbose --top-level cnn -L --test-dir $SDK_TARGET --prefix cats-dogs --checkpoint-file trained/ai85-catsdogs-chw.pth.tar --config-file networks/cats-dogs-chw.yaml --device "$DEVICE" --compact-data --mexpress --softmax --display-checkpoint --embedded-code $EXTRA_ARGS "$@"
