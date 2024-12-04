#!/bin/bash

python3 ../train.py                 \
-model_name=youtube_ffeg2is         \
-year=2018                          \
-dataset=youtubevos                 \
-batch_size=4                       \
-length_clip=3                      \
-max_epoch=5                        \
--resize                            \
-gpu_id=0                           \
-lr_cnn=1e-2                        \
-lr=1e-2                            \
-lr_cva=1e-2                        \
-optim=sgd                          \
-optim_cnn=sgd                      \
-weight_decay=1e-4                  \
-weight_decay_cnn=1e-4              \
--accumulation_steps=8



#python3 ../src/train.py -model_name=/media/zhangying/Datas/gitCode/MyModel/src/models/davis_opcp -year=2016 \
#-dataset=davis -batch_size=6 -length_clip=3 -max_epoch=50 --resize -gpu_id=0  \
#-lr_cnn=1e-3 -lr=1e-2 -lr_cva=1e-2  -optim=sgd -optim_cnn=sgd \
#-weight_decay=1e-4 -weight_decay_cnn=1e-4 --accumulation_steps=1


