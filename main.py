# -*-coding:utf-8 -*-

import pandas as pd
import tensorflow as tf
from tensorflow.contrib import rnn
import opts
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.cross_validation import train_test_split
import pickle
import argparse
import sys
reload(sys)
sys.setdefaultencoding('utf-8')


def generate_batch(x, y, n_example, batch_size):
	for batch_i in xrange(n_example // batch_size):
		start = batch_i * batch_size
		end = start + batch_size
		batch_x = x[start: end]
		batch_y = y[start: end]
		
		yield batch_x, batch_y


