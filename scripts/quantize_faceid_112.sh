#!/bin/sh
python quantize.py trained/ai85-faceid_112-qat.pth.tar trained/ai85-faceid_112-qat-q.pth.tar --device MAX78000 -v "$@"
