#!/bin/sh
python train.py --deterministic --regression --print-freq 1 --epochs 400 --optimizer Adam --lr 0.001 --wd 0 --model ai85autoencoder_ecg --use-bias --dataset SampleECG_ForTrain --device MAX78000 --batch-size 32 --validation-split 0 --show-train-accuracy full --qat-policy policies/qat_policy_autoencoder_ecg.yaml --save-sample 10 "$@"
