'''
using logisticRegression method in the sklearn library for implementing logistic regression
data: classification.txt first 3 columns
target data: classification.txt 5th columns

#Group Member:
#Bin Zhang (5660329599)
#Yihang Chen (6338254416)
#Hanzhi Zhang (4395561906)

'''
import numpy as np
import matplotlib.pyplot as plt

from sklearn import linear_model


def loaddata(path):

	with open(path, 'r') as f:

		data = ';'.join(f.readlines())

		return np.mat(data)
	pass



if __name__ == '__main__':
	
	path = 'classification.txt'
	data = loaddata(path)
	y = np.array(list(map(lambda x:x[0,0], data[:,4])))
	# print(y)
	regr = linear_model.LogisticRegression(max_iter = 7000)
	regr.fit(data[:,:3], y)
	print('Coefficients: \n', regr.coef_)
	print('Intercept:n', regr.intercept_)
	print('scores:', regr.score(data[:,:3], y))







