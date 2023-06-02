#!/bin/sh
DEVICE="MAX78002"
TARGET="sdk/Examples/$DEVICE/CNN"
COMMON_ARGS="--device $DEVICE --timer 0 --display-checkpoint --verbose"

# tests all regression network's first resolution outputs
python ai8xize.py --no-unload --stop-after 88 --max-verify-length 5000 --config-file networks/ai87-pascalvoc-fpndetector-for-kat3.yaml --test-dir $TARGET --prefix pascalvoc_fpndetector_KAT3 --checkpoint-file trained/ai87-pascalvoc-fpndetector-qat8-q.pth.tar --fifo --sample-input tests/sample_pascalvoc_256_320.npy --overwrite $COMMON_ARGS "$@"
