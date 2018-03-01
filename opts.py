# -*- coding:utf-8 -*-
import numpy as np


def open_txt(file_path):
	with open(file_path, 'r+') as f:
		while True:
			line = f.readline()
			if not line:
				return
			yield line.strip()


def train_data(file_path, ndim):
	x = []
	y = []
	for text in open_txt(file_path):
		line = text.split(',')
		y.append(int(line[1]))
		tmpx = np.array([0] * ndim)
		line = text.split('[')
		if len(line[1]) > 1:
			tmpx = line[1].replace(']', '').split(',')
			tmpx = [int(value) for value in tmpx]
		x.append(tmpx)
	return x, y


def score_data(file_path, ndim):
	x = []
	for text in open_txt(file_path):
		line = text.split(",")
		tmpx = np.array([0] * ndim)
		line = text.split("[")
		if len(line[1]) > 1:
			tmpx = line[1].replace(']', '').split(',')
			tmpx = [int(value) for value in tmpx]
		x.append(tmpx)
	return x


def list2txt(List, file_name):
	f = open(file_name, "wb")
	for item in List:
		line = str(round(item, 8)) + "\n"
		f.write(line)
	f.close()

	
