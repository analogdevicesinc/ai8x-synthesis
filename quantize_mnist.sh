#!/bin/sh
./quantize.py trained/ai85-mnist-unquantized.pth.tar trained/ai85-mnist.pth.tar --device MAX78000 -v -c networks/mnist-chw-ai85.yaml --scale 0.85 $@
