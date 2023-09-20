#!/bin/sh
DEVICE="MAX78000"
TARGET="sdk/Examples/$DEVICE/CNN"
COMMON_ARGS="--device $DEVICE --timer 0 --display-checkpoint --verbose"

python ai8xize.py --test-dir $TARGET --prefix bayer2rgb --checkpoint-file trained/ai85-b2rgb-qat8-q.pth.tar --config-file networks/ai85-bayer2rgb.yaml --fifo --sample-input tests/sample_imagenet_bayer.npy $COMMON_ARGS "$@"
