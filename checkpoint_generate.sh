#!/bin/sh

CUDA_VISIBLE_DEVICES=1 python gschnet_script.py \
    train gschnet data models/gschnet-100k-10k \
    --split 100000 10000 --cuda
    # train gschnet data models/gschnet-edm \
    # --split_path=edm_splits.npz --remove_invalid False --cuda

CUDA_VISIBLE_DEVICES=1 python generation.py --epoch -1 --num_mols=10000 --chunk_size=1000
