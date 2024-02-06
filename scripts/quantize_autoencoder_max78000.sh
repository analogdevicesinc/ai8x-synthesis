#!/bin/sh
python quantize.py trained/ai85-autoencoder-samplemotordatalimerick-qat.pth.tar trained/ai85-autoencoder-samplemotordatalimerick-qat-q.pth.tar --device MAX78000 "$@"
