#!/bin/bash

dpath='/home/zuning.duan/adta'

# Train 
#python main.py -model train -dpath ${dpath}/cnn_data_12x24.csv -ndim 288 -ckpt model_rnn.ckpt

# Score
python main.py -model score -dpath ${dpath}/cnn_data_12x24.csv -ndim 288 -ckpt model_rnn.ckpt

