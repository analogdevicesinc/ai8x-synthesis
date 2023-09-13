#!/bin/sh
python quantize.py trained/ai85-facedet-tinierssd-qat8.pth.tar trained/ai87-facedet-tinierssd-qat8-q.pth.tar --device MAX78002 -v "$@"
