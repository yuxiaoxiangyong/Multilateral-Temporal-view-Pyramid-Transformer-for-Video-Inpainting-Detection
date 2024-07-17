#!/bin/bash

#python3 ../src/test.py -model_name=youtube_ffeg2pp -dataset=youtubevos -eval_split=val -batch_size=1  -gpu_id=0 -num_workers=4 -year=2018

python3 ../src/test.py -model_name=youtube_viop -dataset=davis -eval_split=val -batch_size=1  -gpu_id=0 -num_workers=1 -year=2016