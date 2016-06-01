#################################################
# logRegression: Logistic Regression
# Author : Yansong Gao, Jingyuan Li
# Date   : 2016-06-01
# Email  : yag037@ucsd.edu
#################################################

from numpy import *
import matplotlib.pyplot as plt
import time
import generate_sample as gs

MEAN = 0.25
MAX_ITR = 10
ALPHA = 0.005
TESTSET_SIZE = 400
SCENARIO = 'hyperball'
# calculate the sigmoid function
def sigmoid(inX):
	return 1.0 / (1 + exp(-inX))

def loss_gradient(w, x, y):
    x_tilde = hstack((x, [1]))
    return -y * exp(-y * w.dot(x_tilde)) * x_tilde / (1 + exp(-y * w.dot(x_tilde)))


def loss(weights, inX, inY):
	return log(1+exp(-inX * inX * weights))


# train a logistic regression model using SGD
# input: train_x is a mat datatype, each row stands for one sample
#		 train_y is mat datatype too, each row is the corresponding label
#		 alpha is the step size
#        maxIter is the number of iterations
def trainLogRegres(train_x, train_y, alpha, maxIter, scenario):
	# calculate training time
	startTime = time.time()

	numSamples, numFeatures = shape(train_x)
	# weights = ones(numFeatures)
	weights = zeros((n+1, 5))
	# w = np.zeros((n + 1, DIM))

	for t in range(numSamples):
        # alpha = step_size(t + 1, scenario, n)
        G = loss_gradient(w[t], train_x, train_y)
     
        weights[t + 1] = gs.projection(scenario ,w[t] - alpha * G)
        
    return np.mean(weights[1:], axis = 0)


	for k in range(maxIter):
		for i in range(numSamples):
			output = sigmoid(train_x[i] * weights)
			error = train_y[i] - output
			weights = weights + alpha * train_x[i].transpose() * error
	print 'Congratulations, training complete! Took %fs!' % (time.time() - startTime)
	return weights


# test your trained Logistic Regression model given test set
def testLogRegres(weights, test_x, test_y):
	numSamples, numFeatures = shape(test_x)
	matchCount = 0
	errorCount = 0
	totalLoss = 0
	for i in xrange(numSamples):
		true_result = False
		if test_y[i] == -1:
			true_result = False
		else:
			true_result = True
		predict = sigmoid(inner(test_x[i] , weights)) > 0.5
		
		if predict == true_result:
			matchCount += 1
		else:
			errorCount += 1
	accuracy = float(matchCount) / numSamples
	errorRate = float(errorCount) / numSamples
	totalLoss = gs.loss(weights, test_y, test_x)
		
	return errorRate, float(sum(totalLoss)) / numSamples 


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



def main():
	n = [50, 100, 500, 1000]
	sigma = [0.05, 0.25]
	for sub_sigma in sigma:
		expected_err = []
		expected_risk = []
		for sub_n in n:
			## generate test set
			test_x, test_y = gs.generate_samples(TESTSET_SIZE, MEAN, sub_sigma, SCENARIO)
			true_err_exp = []
			risk_exp = []
			for i in range(20):
				## generate training set
				train_x, train_y = gs.generate_samples(sub_n, MEAN, sub_sigma, SCENARIO)
				learned_weights = trainLogRegres(train_x, train_y, ALPHA, MAX_ITR)
				
				true_err, risk = testLogRegres(learned_weights, test_x, test_y)
				true_err_exp.append(true_err)
				risk_exp.append(risk)
			expected_err.append(sum(true_err_exp)/20)
			expected_risk.append(sum(risk_exp)/20)
		print "scenario: ", SCENARIO, "sigma = ", sub_sigma, "expected true err:", expected_err, "expected risk:", expected_risk
		## plot two figures here

if __name__ == '__main__':
	main()






