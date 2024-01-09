#!/bin/sh
DEVICE="MAX78002"
TARGET="sdk/Examples/$DEVICE/CNN"
COMMON_ARGS="--device $DEVICE --timer 0 --display-checkpoint --verbose"

python ai8xize.py --test-dir $TARGET --prefix mobilefacenet-112 --checkpoint-file trained/ai87-mobilefacenet-112-qat-q.pth.tar --config-file networks/ai87-mobilefacenet-112.yaml --fifo $COMMON_ARGS "$@"
