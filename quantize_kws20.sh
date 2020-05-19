#!/bin/sh
./quantize.py trained/ai85-kws20-unquantized.pth.tar trained/ai85-kws20.pth.tar --device 85 -v -c networks/kws20-hwc.yaml --scale 0.97
