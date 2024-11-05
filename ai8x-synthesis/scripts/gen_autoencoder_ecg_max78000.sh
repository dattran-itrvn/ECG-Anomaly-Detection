#!/bin/sh
DEVICE="MAX78000"
TARGET="/home/dattran/MaximSDK/Examples/$DEVICE/CNN"
COMMON_ARGS="--device $DEVICE --timer 0 --display-checkpoint --verbose"

python ai8xize.py --test-dir $TARGET --prefix autoencoder_ecg --checkpoint-file trained/ai85-autoencoder-ecg-qat-q.pth.tar --config-file networks/ai85-autoencoder-ecg.yaml --sample-input tests/sample_sampleecg_forevalwithsignal.npy --energy  $COMMON_ARGS "$@"
