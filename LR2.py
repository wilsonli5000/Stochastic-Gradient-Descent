#################################################
# logRegression: Logistic Regression
# Author : Yansong Gao, Jingyuan Li
# Date   : 2016-06-01
# Email  : yag037@ucsd.edu
#################################################

from numpy import *
import matplotlib.pyplot as plt
import time


# calculate the sigmoid function
def sigmoid(inX):
	return 1.0 / (1 + exp(-inX))


# train a logistic regression model using SGD
# input: train_x is a mat datatype, each row stands for one sample
#		 train_y is mat datatype too, each row is the corresponding label
def trainLogRegres(train_x, train_y):
	# calculate training time
	startTime = time.time()

	numSamples, numFeatures = shape(train_x)
	alpha = opts['alpha']; maxIter = opts['maxIter']
	weights = ones((numFeatures, 1))

	# optimize through gradient descent algorilthm
	for k in range(maxIter):
		for i in range(numSamples):
			output = sigmoid(train_x[i, :] * weights)
			error = train_y[i, 0] - output
			weights = weights + alpha * train_x[i, :].transpose() * error
	print 'Congratulations, training complete! Took %fs!' % (time.time() - startTime)
	return weights


# test your trained Logistic Regression model given test set
def testLogRegres(weights, test_x, test_y):
	numSamples, numFeatures = shape(test_x)
	matchCount = 0
	for i in xrange(numSamples):
		predict = sigmoid(test_x[i, :] * weights)[0, 0] > 0.5
		if predict == bool(test_y[i, 0]):
			matchCount += 1
	accuracy = float(matchCount) / numSamples
	return accuracy


# show your trained logistic regression model only available with 2-D data
def showLogRegres(weights, train_x, train_y):
	# notice: train_x and train_y is mat datatype
	numSamples, numFeatures = shape(train_x)
	if numFeatures != 3:
		print "Sorry! I can not draw because the dimension of your data is not 2!"
		return 1

	# draw all samples
	for i in xrange(numSamples):
		if int(train_y[i, 0]) == 0:
			plt.plot(train_x[i, 1], train_x[i, 2], 'or')
		elif int(train_y[i, 0]) == 1:
			plt.plot(train_x[i, 1], train_x[i, 2], 'ob')

	# draw the classify line
	min_x = min(train_x[:, 1])[0, 0]
	max_x = max(train_x[:, 1])[0, 0]
	weights = weights.getA()  # convert mat to array
	y_min_x = float(-weights[0] - weights[1] * min_x) / weights[2]
	y_max_x = float(-weights[0] - weights[1] * max_x) / weights[2]
	plt.plot([min_x, max_x], [y_min_x, y_max_x], '-g')
	plt.xlabel('X1'); plt.ylabel('X2')
	plt.show()