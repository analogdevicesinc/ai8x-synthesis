#!/bin/sh
python quantize.py trained/ai87-mobilefacenet-112-qat.pth.tar trained/ai87-mobilefacenet-112-qat-q.pth.tar --device MAX78002 -v "$@"
