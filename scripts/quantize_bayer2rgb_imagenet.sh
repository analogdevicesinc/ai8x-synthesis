#!/bin/sh
python quantize.py trained/ai85-bayer2rgb-qat8.pth.tar trained/ai85-bayer2rgb-qat8-q.pth.tar --device MAX78000 "$@"
