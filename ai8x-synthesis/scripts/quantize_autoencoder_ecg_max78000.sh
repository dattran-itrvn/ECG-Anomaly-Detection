#!/bin/sh
python quantize.py trained/ai85-autoencoder-ecg-qat.pth.tar trained/ai85-autoencoder-ecg-qat-q.pth.tar --device MAX78000 "$@"
