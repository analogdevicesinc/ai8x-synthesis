#!/bin/sh
./quantize.py trained/ai85-faceid-unquantized.pth.tar trained/ai85-faceid.pth.tar --device MAX78000 -v -c networks/faceid.yaml --scale 1.05 "$@"
