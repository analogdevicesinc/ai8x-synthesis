#!/bin/sh
python quantize.py trained/ai85-kws20_v3_msnoise_mixed-qat8.pth.tar trained/ai85-kws20_v3_msnoise_mixed-qat8-q.pth.tar --device MAX78000 -v "$@"
