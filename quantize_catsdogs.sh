#!/bin/sh
./quantize.py trained/ai85-catsvsdogs-unquantized.pth.tar trained/ai85-catsdogs-chw.pth.tar --device MAX78000 -v -c networks/cats-dogs-chw.yaml --scale 0.85 "$@"
