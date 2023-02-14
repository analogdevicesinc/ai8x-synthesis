#!/bin/sh
python3 quantize.py trained/ai87-kws20_v3-qat8.pth.tar trained/ai87-kws20_v3-qat8-q.pth.tar --device MAX78002 -v "$@"
