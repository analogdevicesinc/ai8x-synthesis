#!/bin/sh
python quantize.py trained/ai87-cifar100-effnet2-qat8.pth.tar trained/ai87-cifar100-effnet2-qat8-q.pth.tar --device MAX78002 -v -c networks/ai87-cifar100-effnet2.yaml "$@"
