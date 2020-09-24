'''
using linear model methods in the sklearn for linear classification
data: classification.txt first 3 columns
target data: classification.txt 4th columns

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
	

	data = loaddata('classification.txt')
	x = data[:,:3]
	y = np.array(list(map(lambda x:x[0,0], data[:,3])))

	classifier = linear_model.SGDClassifier()
	classifier.fit(x,y)

	print('Coefficients: \n', classifier.coef_)
	print('Intercept:n', classifier.intercept_)
	print('scores:', classifier.score(x, y))



