#!/bin/sh
./quantize.py trained/ai85-cifar100-unquantized.pth.tar trained/ai85-cifar100.pth.tar --device 85 -v -c networks/cifar100-simple.yaml --scale 1.0
