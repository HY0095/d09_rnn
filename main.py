# -*- coding:utf-8 -*-

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


def weight_variable(shape, w_alpha=0.01):
	initial = w_alpha * tf.random_normal(shape)
	return tf.Variable(initial)


def bias_variable(shape, b_alpha=0.1):
	initial = b_alpha * tf.random_normal(shape)
	return tf.Variable(initial)


def model(x, weight, biases, lstm_size, input_vec_size, time_step_size):

	# X, input shape: (batch_size, time_step_size, input_vec_size)
	# Xt shape: (time_step_size, batch_size, input_vec_size)
	xt = tf.transpose(x, [1, 0, 2]) # permute time_step_size and batch_size, [28, ?, 28]

	# Xt shape: (time_step_size * batch_size * input_vec_size)
      	xr = tf.reshape(xt, [-1, input_vec_size]) # each row has input for each cell (lstm_size = input_vec_size)

	# Each array shape: (batch_size, input_vec_size)
	x_split = tf.split(xr, time_step_size) # split them into time_step_size, shape=[(128, 28), (128, 28), ...]

	# Make lstm with lstm_size(each input vector size). num_units=lstm_size; forget_bias=1.0
	lstm = rnn.BasicLSTMCell(lstm_size, forget_bias=1.0, state_is_tuple=True)
	
	# Get lstm cell output, time_step_size (12) arrays with lstm_size output: (batch_size, lstm_size)
	# rnn.static_rnn()的输出对应每一个timestep, 如果只关心最后一部的输出，取outputs[-1]即可
	outputs, states = rnn.static_rnn(lstm, x_split, dtype=tf.float32)
	
	# Liner activation, using rnn inner loop last output
	return tf.matmul(outputs[-1], weights) + biases


def build_rnn(x_data, y_data, ratio, ckpt):

	# Save MinMaxScaler as pickle file
	x_scaler = np.array(x_data).astype(np.float32)
	scaler = MinMaxScaler()
	scaler.fit(x_scaler)
	f = open("MinMaxScaler.pkl", 'w+')
	pickel.dump(scaler, f)
	f.close()

	x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=ratio, random_state=123)

	y_train = np.array(y_train).astype(np.float32).reshape(-1, 1)
	y_test = np.array(y_test).astype(np.float32).reshape(-1, 1)
	x_train = np.array(x_train).astype(np.float32)
	x_test = np.array(x_test).astype(np.float32)

	# Scaler = MinMaxScaler()
	x_train = scaler.transform(x_train)
	x_test = scaler.transform(x_test)
	X_train = x_train.reshape(-1, 12, 24)
	X_test = x_test.reshape(-1, 12, 24)
	Y_train = OneHotEncoder().fit_transform(y_train).todense()
	Y_test = OneHotEncoder().fit_transform(y_test).todense()

	# Training Parameters
	learning_tate = 0.001
	batch_size = 128
	
	# NetWork Parameters
	input_vec_size = 24
	time_step_size = 12
	lstm_size = 256 # LSTM 隐藏神经元数量
	num_classes = 2

	# tf graph input
	tf_x = tf.placeholder(tf.float32, [None, time_step_size, input_vec_size])
	tf_y = tf.placeholder(tf.float32, [None, num_classes])

	# Define weights
	weights = weight_variable([lstm_size, num_classes])
	biases = bias_variable([num_classes])

	logits = model(tf_x, weights, biases, lstm_size, input_vec_size, time_step_size)
	prediction = tf.nn.softmax(logits)

	# Define loss and optimizer
	# cross_entropy = -tf.reduce_sum(tf_y * tf.log(prediction))
	# train_op = tf.train.AdamOptimizer(1e-4).minimize(loss_op)

	loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf_y))
	# optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
	train_op = tf.train.AdamOptimizer(1e-4).minimize(loss_op)

	# Evaluate model (with test logits, for dropout to be disabled)
	correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(tf_y, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

	# Initialize the variable (i.e. assign their default value)
	init = tf.global_variables_initializer()

	# Save Model
	saver = tf.train.Saver()

	# Start training
	with tf.Session() as sess:
		# Run the initializer
		sess.run(init)
		
		for epoch in xrange(9000):
			for batch_x, batch_y in generate_batch(X_train, Y_train, Y_train.shape[0], batch_size):
				# Run optimization op (backprob)
				sess.run(train_op, feed_dict={tf_x: batch_x, tf_y: batch_y})
			if epoch % 100 == 0:
				# Calculate batch loss and accuracy
				loss, acc = sess.run([loss_op, accuracy], feed_dict{tf_x: X_train, tf_y: Y_train})
				print "Step: %d Minibatch Loss = %.4f, Training Accuracy = %.6f" % (epoch, loss, acc)
		print "Optimization Finished"
		acc = sess.run(accuracy, feed_dict={tf_x: X_test, tf_y: Y_test})
		print "Testing Accuracy = %.6f" % acc

		# Save Rnn-Model
		save_path = saver.save(sess, ckpt)


def score(estimator, scaler, x_data):

	result = []
	# Load Scaler
	fr = open("MinMaxScaler.pkl", 'rb')
	scaler = pickle.load(fr)
	fr.close()

	# Data Transform
	x_data = np.array(x_data).astype(np.float32)
	x_data = scaler.transform(x_data)
	x_data = x_data.reshape(-1, 12, 24)

	# Network Parameters
	input_vec_size = 24
	time_step_size = 12
	lstm_size = 256 # LSTM 隐藏神经元数量
	num_classes = 2

	# tf Graph input
	tf_x = tf.placeholder(tf.float32, [None, time_step_size, input_vec_size])
	tf_y = tf.placeholder(tf.float32, [None, num_classes])

	# Define weights
	weights = weight_variable([lstm_size, num_classes])
	biases = bias_variable([num_classes])

	logits = model(tf_x, weights, biases, lstm_size, input_vec_size, time_step_size)
	prediction = tf.nn.softmax(logits)

	saver = tf.train.Saver()
	# Score
	with tf.Session() as sess:
		# saver = tf.train.import_meta_graph('model_cnn.ckpt.meta')
		# saver.restore(sess, tf.train.latest_checkpoint('./model.ckpt'))
		saver.restore(sess, './model_rnn.ckpt')
		result = sess.run(prediction, feed_dict={tf_x: x_data})

	opts.list2txt(result[:, 1], 'score.txt')
	return result


if __name__ = "__main__":
	
	parser = argparse.ArgumentParser()
	parser.add_argument("-model", default="train", help="train || score")
	parser.add_argument("-dpath", default='/home/zuning.duan/model/d03_xgboost/data/cnn_data.csv')
	parser.add_argument("-ckpt", default="/home/zuning.duan/rnn/model.ckpt")
	parser.add_argument("-scaler", default="/home/zuning.duan/rnn/MinMaxScaler.pkl")
	parser.add_argument("-ndim", default=288)
	parser.add_argument("-ratio", default=0.4)
	args = parser.parse_args()

	if args.model == "train":
		
		x_data, y_data = opts.train_data(args.dpath, int(args.ndim))
		print "==== Model: %s ==== " % args.model
		build_rnn(x_data, y_data, args.ratio, args.ckpt)
	
	else args.model == "score":
		
		x_data = opts.score_data(args.dpath, int(args.ndim))
		print "==== Model: %s ====" % args.model
		result = socre(args.ckpt, args.scaler, x_data)	


