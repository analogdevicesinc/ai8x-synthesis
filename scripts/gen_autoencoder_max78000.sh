#!/bin/sh
DEVICE="MAX78000"
TARGET="sdk/Examples/$DEVICE/CNN"
COMMON_ARGS="--device $DEVICE --timer 0 --display-checkpoint --verbose"

python ai8xize.py --test-dir $TARGET --prefix autoencoder --checkpoint-file trained/ai85-autoencoder-samplemotordatalimerick-qat-q.pth.tar --config-file networks/ai85-autoencoder.yaml --sample-input tests/sample_motordatalimerick_fortrain.npy --energy  $COMMON_ARGS "$@"
