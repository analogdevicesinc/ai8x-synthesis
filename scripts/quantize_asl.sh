#!/bin/sh
./quantize.py ../ai8x-training/logs/2021.07.23-131712/ai85-asl-qat8.pth.tar trained/ai85-asl-qat8-q.pth.tar --device MAX78000 -v -c networks/asl-chw.yaml "$@"


