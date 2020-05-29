#!/bin/sh
./quantize.py trained/ai85-cifar10-unquantized.pth.tar trained/ai85-cifar10.pth.tar --device 85 -v -c networks/cifar10-hwc-ai85.yaml
