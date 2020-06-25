#!/bin/sh
CLOCK_TRIM=""
# Example: CLOCK_TRIM="--clock-trim 0x173,0x5e,0x14"
SDK_TARGET="sdk/Examples/MAX78000/CNN"

# Update common code and header files
mkdir -p $SDK_TARGET/Common/
cp device-ai85/softmax.c $SDK_TARGET/Common/
cp device-ai85/*.h $SDK_TARGET/Common/

./ai8xize.py -e --verbose --top-level cnn -L --test-dir $SDK_TARGET --prefix cifar-100-ressimplenet --checkpoint-file trained/ai85-cifar100-ressimplenet.pth.tar --config-file networks/cifar100-ressimplenet.yaml --device 85 --compact-data --mexpress --display-checkpoint $CLOCK_TRIM $@
