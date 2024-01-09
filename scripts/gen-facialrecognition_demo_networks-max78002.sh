#!/bin/sh
DEVICE="MAX78002"
TARGET="sdk/Examples/$DEVICE/CNN"
COMMON_ARGS="--device $DEVICE --timer 0 --display-checkpoint --verbose"

python ai8xize.py --test-dir $TARGET --prefix mobilefacenet_demo --checkpoint-file trained/ai87-mobilefacenet-112-qat-q.pth.tar --config-file networks/ai87-mobilefacenet-112.yaml --fifo $COMMON_ARGS "$@"
python ai8xize.py --test-dir $TARGET --prefix dotprod_demo --checkpoint-file trained/ai87-dot_emb_112_checkpoint-q.pth.tar --config-file networks/ai87-dotprod.yaml --start-layer 73 --weight-start 2000 $COMMON_ARGS "$@"
python ai8xize.py --test-dir $TARGET --prefix facedetection_demo --checkpoint-file trained/ai87-facedet-tinierssd-qat8-q.pth.tar --config-file networks/ai87-face-tinierssd-startat74.yaml --start-layer 74 --weight-start 2500 $COMMON_ARGS "$@"
