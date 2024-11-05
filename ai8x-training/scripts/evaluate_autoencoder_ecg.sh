#!/bin/sh
python train.py --deterministic --model ai85autoencoder_ecg --dataset SampleECG_ForEvalWithSignal --regression --device MAX78000 --qat-policy policies/qat_policy_autoencoder_ecg.yaml --use-bias --evaluate --exp-load-weights-from ../ai8x-synthesis/trained/ai85-autoencoder-ecg-qat-q.pth.tar -8 --print-freq 1 --save-sample 10 "$@"
