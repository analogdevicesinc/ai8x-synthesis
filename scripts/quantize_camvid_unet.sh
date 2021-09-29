#!/bin/sh
python3 quantize.py trained/ai85-camvid-unet-large.pth.tar trained/ai85-camvid-unet-large-q.pth.tar --device MAX78000 -v
