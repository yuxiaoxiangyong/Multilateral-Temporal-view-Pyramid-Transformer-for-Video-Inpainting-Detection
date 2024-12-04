#!/bin/bash

# youtube
python3 ../test.py -model_name=mumpy_test -dataset=youtubevos -eval_split=test -batch_size=1  -gpu_id=0 -num_workers=4 -year=2018 -test_epoch=10

# davis
#python3 ../test.py -model_name=mumpy_test -dataset=davis -eval_split=test -batch_size=1 -gpu_id=0 -num_workers=1 -year=2016 -test_epoch=10