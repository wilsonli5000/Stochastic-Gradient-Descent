__author__ = 'Jingyuan Li'

import numpy as np

def generate_samples(sample_size, mean, std, scenario):
	y = np.sign(np.random.uniform(-0.25, 0.25, sample_size))
	x = []
	for label in y:
		x.append(np.random.normal(mean * label, std, 4))
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
			if norm > 1:
				samples = np.divide(samples , np.empty(4).fill(norm))
	return x

def loss(weights, y, x):
	return np.log(1 + np.exp(-y * np.inner(weights, x.append(1.0))))