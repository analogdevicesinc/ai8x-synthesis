#!/bin/sh
DEVICE="MAX78000"
TARGET="sdk/Examples/$DEVICE/CNN"
COMMON_ARGS="--device $DEVICE --timer 0 --display-checkpoint --verbose"

python izer/add_fake_passthrough.py --input-checkpoint-path trained/ai85-faceid_112-qat-q.pth.tar --output-checkpoint-path trained/ai85-fakepass-faceid_112-qat-q.pth.tar --layer-name fakepass --layer-depth 128 --layer-name-after-pt linear --low-memory-footprint "$@"
python ai8xize.py --test-dir $TARGET --prefix faceid_112 --checkpoint-file trained/ai85-fakepass-faceid_112-qat-q.pth.tar --config-file networks/ai85-faceid_112.yaml --fifo $COMMON_ARGS "$@"
