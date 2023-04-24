#!/bin/sh
DEVICE="MAX78002"
TARGET="sdk/Examples/$DEVICE/CNN"
COMMON_ARGS="--device $DEVICE --timer 0 --display-checkpoint --verbose"

python ai8xize.py --no-kat --no-unload --energy --test-dir $TARGET --prefix pascalvoc_fpndetector_pow_mon --checkpoint-file trained/ai87-pascalvoc-fpndetector-qat8-q.pth.tar --config-file networks/ai87-pascalvoc-fpndetector.yaml --fifo --sample-input tests/sample_pascalvoc_256_320.npy --overwrite $COMMON_ARGS "$@"
