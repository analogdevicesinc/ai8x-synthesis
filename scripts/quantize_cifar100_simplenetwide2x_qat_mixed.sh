#!/bin/sh
python quantize.py trained/ai85-cifar100-simplenetwide2x-qat-mixed.pth.tar trained/ai85-cifar100-simplenetwide2x-qat-mixed-q.pth.tar --device MAX78000 -v -c networks/cifar100-simplewide2x.yaml "$@"
