#!/bin/sh
#./quantize.py ../ai8x-training/logs/2021.07.09-172633/checkpoint.pth.tar trained/ai85-asl01-chw.pth.tar --device MAX78000 -v -c networks/asl-chw.yaml "$@"

./quantize.py ../ai8x-training/logs/2021.08.06-164826/qat_best.pth.tar trained/ai85-asl01-chw.pth.tar --device MAX78000 -v -c networks/asl-chw.yaml "$@"


