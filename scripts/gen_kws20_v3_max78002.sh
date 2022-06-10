#!/bin/sh
DEVICE="MAX78002"
TARGET="sdk/Examples/$DEVICE/CNN"
COMMON_ARGS="--device $DEVICE --timer 0 --display-checkpoint --verbose"

python ai8xize.py --test-dir $TARGET --prefix kws20_v3_1 --checkpoint-file trained/ai87-kws20_v3-qat8-q.pth.tar --config-file networks/ai87-kws20-v3-hwc.yaml --softmax $COMMON_ARGS "$@"
