#!/bin/sh
python quantize.py trained/ai85-camvid-unet-large.pth.tar trained/ai85-camvid-unet-large-q.pth.tar --device MAX78000 -v
python izer/add_fake_passthrough.py --input-checkpoint-path trained/ai85-camvid-unet-large-q.pth.tar --output-checkpoint-path trained/ai85-camvid-unet-large-fakept-q.pth.tar --layer-name pt --layer-depth 56 --layer-name-after-pt upconv3
