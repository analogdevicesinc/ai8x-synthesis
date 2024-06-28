#!/bin/sh
python quantize.py trained/ai85-kws20_nas_msnoise_mixed-qat8.pth.tar trained/ai85-kws20_nas_msnoise_mixed-qat8-q.pth.tar --device MAX78000 -v "$@"
