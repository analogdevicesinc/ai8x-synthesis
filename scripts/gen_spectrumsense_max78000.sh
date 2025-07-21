#!/bin/sh
DEVICE="MAX78000"
TARGET="sdk/Examples/$DEVICE/CNN"
COMMON_ARGS="--device $DEVICE --timer 0 --display-checkpoint --verbose"

python ai8xize.py --test-dir $TARGET --prefix spectrumsense --checkpoint-file trained/ai85-spectrumsense-unet-large-fakept-q.pth.tar --config-file networks/spectrumsense-unet-large-fakept.yaml --sample-input tests/sample_spectrumsense_352.npy --energy  $COMMON_ARGS --compact-data --mexpress --overlap-data --mlator --no-unload  "$@"

