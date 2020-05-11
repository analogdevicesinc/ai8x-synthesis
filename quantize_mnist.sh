#!/bin/sh
./quantize.py trained/ai85-mnist-unquantized.pth.tar trained/ai85-mnist.pth.tar --device 85 -v -c networks/mnist-chw-ai85.yaml
