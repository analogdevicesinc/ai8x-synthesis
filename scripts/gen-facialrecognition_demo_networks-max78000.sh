#!/bin/sh
DEVICE="MAX78000"
TARGET="sdk/Examples/$DEVICE/CNN"
COMMON_ARGS="--device $DEVICE --timer 0 --display-checkpoint --verbose"

python izer/add_fake_passthrough.py --input-checkpoint-path trained/ai85-faceid_112-qat-q.pth.tar --output-checkpoint-path trained/ai85-fakepass-faceid_112-qat-q.pth.tar --layer-name fakepass --layer-depth 128 --layer-name-after-pt linear --low-memory-footprint "$@"
python ai8xize.py --test-dir $TARGET --prefix faceid_112 --checkpoint-file trained/ai85-fakepass-faceid_112-qat-q.pth.tar --config-file networks/ai85-faceid_112.yaml --fifo $COMMON_ARGS "$@"
python ai8xize.py --test-dir $TARGET --prefix dotprod_demo --checkpoint-file trained/ai85-dot_emb_112_checkpoint-q.pth.tar --config-file networks/ai85-dotprod.yaml $COMMON_ARGS "$@"
python ai8xize.py --test-dir $TARGET --prefix facedet_tinierssd --checkpoint-file trained/ai85-facedet-tinierssd-qat8-q.pth.tar --config-file networks/ai85-facedet-tinierssd.yaml --fifo  $COMMON_ARGS "$@"
