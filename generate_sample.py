__author__ = 'Jingyuan Li'

import numpy as np

SAMPLE_SIZE = 500
STD = 0.25
MEAN = 0.25
y = np.sign(np.random.uniform(-0.25, 0.25, SAMPLE_SIZE))
x = []
for label in y:
	x.append(np.random.normal(MEAN * label, STD, 4))

print (x)