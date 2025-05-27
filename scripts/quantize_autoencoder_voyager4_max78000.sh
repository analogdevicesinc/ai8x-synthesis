#!/bin/sh
python quantize.py trained/ai85-autoencoder-motordatavoyager4-qat.pth.tar trained/ai85-autoencoder-motordatavoyager4-qat-q.pth.tar --device MAX78000 "$@"

