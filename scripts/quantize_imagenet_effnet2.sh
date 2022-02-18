#!/bin/sh
python quantize.py trained/ai87-imagenet-effnet2.pth.tar trained/ai87-imagenet-effnet2-q.pth.tar --device MAX78002 -v -c networks/ai87-imagenet-effnet2.yaml "$@"
