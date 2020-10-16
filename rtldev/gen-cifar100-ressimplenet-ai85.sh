#!/bin/sh
DEVICE="MAX78000"
SDK_TARGET="sdk/Examples/$DEVICE/CNN"
EXTRA_ARGS=""
# Example: EXTRA_ARGS="--boost 2.5"

# Update common code and header files
mkdir -p $SDK_TARGET/Common/
cp assets/device-ai85/softmax.c $SDK_TARGET/Common/
cp assets/device-ai85/*.h $SDK_TARGET/Common/

./ai8xize.py -e --verbose --top-level cnn -L --test-dir $SDK_TARGET --prefix cifar-100-ressimplenet --checkpoint-file trained/ai85-cifar100-ressimplenet.pth.tar --config-file networks/cifar100-ressimplenet.yaml --device "$DEVICE" --compact-data --mexpress --display-checkpoint $EXTRA_ARGS "$@"
