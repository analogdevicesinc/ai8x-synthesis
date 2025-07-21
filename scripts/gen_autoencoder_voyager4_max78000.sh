#!/bin/sh
DEVICE="MAX78000"
TARGET="sdk/Examples/$DEVICE/CNN"
COMMON_ARGS="--device $DEVICE --timer 0 --display-checkpoint --verbose"

python ai8xize.py --test-dir $TARGET --prefix autoencoder_voyager4 --checkpoint-file trained/ai85-autoencoder-motordatavoyager4v1-qat-q.pth.tar --config-file networks/ai85-autoencoder.yaml --sample-input tests/sample_motordatavoyager4_fortrain.npy --energy  $COMMON_ARGS "$@"
