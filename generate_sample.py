__author__ = 'Jingyuan Li'

import numpy as np

def generate_samples(sample_size, mean, std, scenario):
	y = np.sign(np.random.uniform(-0.25, 0.25, sample_size))
	x = []
	for label in y:
		x.append(np.append(np.random.normal(mean * label, std, 4), 1.0))
		
	return projection(scenario, x), y


def projection(scenario, x):
	if scenario is 'hypercube':
		for samples in x:
			for dim in samples:
				if dim > 1: dim = 1
				elif dim < -1: dim = -1
	elif scenario is 'hyperball':
		for samples in x:
			norm = np.linalg.norm(samples)
			print np.empty(5).fill(norm)
			if norm > 1:
				samples = np.divide(samples , np.empty(5).fill(norm))
	return x

def loss(weights, y, x):
	loss = []
	for i in range(y.size):
		loss.append(np.log(np.exp(np.inner(weights, x[i]) * (-y[i])) + 1))
	return loss