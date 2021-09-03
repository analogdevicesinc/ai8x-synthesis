#!/bin/sh
./quantize.py trained/ai85-asl-qat8.pth.tar trained/ai85-asl-qat8-q.pth.tar --device MAX78000 -v -c networks/asl-chw.yaml "$@"
