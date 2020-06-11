#!/bin/sh
./quantize.py trained/ai85-mnist-extrasmall-unquantized.pth.tar trained/ai85-mnist-extrasmall.pth.tar --device MAX78000 -v -c networks/mnist-chw-extrasmall-ai85.yaml --scale 0.85 $@
