#!/bin/bash

python3 ../train.py                \
-model_name=mumpy_test             \
-year=2016                         \
-dataset=davis                     \
-batch_size=6                      \
-length_clip=3                     \
-max_epoch=50                      \
--resize                           \
-gpu_id=0                          \
-lr_cnn=1e-3                       \
-lr=1e-2                           \
-lr_cva=1e-2                       \
-optim=sgd                         \
-optim_cnn=sgd                     \
-weight_decay=1e-4                 \
-weight_decay_cnn=1e-4             \
--accumulation_steps=1



