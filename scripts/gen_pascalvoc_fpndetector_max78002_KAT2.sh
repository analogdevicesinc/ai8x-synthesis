#!/bin/sh
DEVICE="MAX78002"
TARGET="sdk/Examples/$DEVICE/CNN"
COMMON_ARGS="--device $DEVICE --timer 0 --display-checkpoint --verbose"

# tests all classification network outputs other than the first resolution
python ai8xize.py --stop-after 79 --no-unload --config-file networks/ai87-pascalvoc-fpndetector-for-kat2.yaml --test-dir $TARGET --prefix pascalvoc_fpndetector_KAT2 --checkpoint-file trained/ai87-pascalvoc-fpndetector-qat8-q.pth.tar --fifo --sample-input tests/sample_pascalvoc_256_320.npy --overwrite $COMMON_ARGS "$@"
