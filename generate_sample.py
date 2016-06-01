__author__ = 'Jingyuan Li'

import numpy as np

def generate_samples(sample_size, mean, std):
	y = np.sign(np.random.uniform(-0.25, 0.25, sample_size))
	x = []
	for label in y:
		x.append(np.random.normal(mean * label, std, 4))
	return x, y

def loss(weights, y, x):
	return np.log(1 + np.exp(-y * np.inner(weights, x.append(1.0))))