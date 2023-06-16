#!/bin/sh
DEVICE="MAX78000"
TARGET="sdk/Examples/$DEVICE/CNN"
COMMON_ARGS="--device $DEVICE --timer 0 --display-checkpoint --verbose"

python ai8xize.py --test-dir $TARGET --prefix facedet_tinierssd --checkpoint-file trained/ai85-facedet-tinierssd-qat8-q.pth.tar --config-file networks/ai85-facedet-tinierssd.yaml --sample-input tests/sample_vggface2_facedetection.npy --fifo  $COMMON_ARGS "$@"