#!/bin/sh
DEVICE="MAX78002"
TARGET="sdk/Examples/$DEVICE/CNN"
COMMON_ARGS="--device $DEVICE --timer 0 --display-checkpoint --verbose"

# --stop-after is used for testing only first resolution output for classification network as all outputs do not fit to flash all together. max-verify-length also should be set not to get flash error for this output
python ai8xize.py --stop-after 52 --no-unload --max-verify-length 20000 --test-dir $TARGET --prefix pascalvoc_fpndetector_KAT1 --checkpoint-file trained/ai87-pascalvoc-fpndetector-qat8-q.pth.tar --config-file networks/ai87-pascalvoc-fpndetector.yaml --fifo --sample-input tests/sample_pascalvoc_256_320.npy --overwrite $COMMON_ARGS "$@"
