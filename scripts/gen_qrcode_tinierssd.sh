#!/bin/sh
DEVICE="MAX78000"
TARGET="sdk/Examples/$DEVICE/CNN"
COMMON_ARGS="--device $DEVICE --timer 0 --display-checkpoint --verbose"

python ai8xize.py --test-dir $TARGET --prefix tinierssd_kpts_qrcode --checkpoint-file trained/ai85-qrcode-tinierssd-kpts-qat8-q.pth.tar --config-file networks/ai85-tinierssd-kpts-qr.yaml --fifo $COMMON_ARGS --synthesize-input 0x2580  "$@"
