'''
using linear regression methods in the sklearn library
in linear-regression.txt
The first 2 columns in each row represent the independent X and Y variables; and the 3rd column represents the dependent Z variable.

#Group Member:
#Bin Zhang (5660329599)
#Yihang Chen (6338254416)
#Hanzhi Zhang (4395561906)

'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import numpy as np


def loaddata(path):

	with open(path, 'r') as f:

		data = ';'.join(f.readlines())

		return np.mat(data)
	pass



if __name__ == '__main__':
	
	path = 'linear-regression.txt'
	data = loaddata(path)
	# print(data[:,0:2])

	regr = linear_model.LinearRegression()
	regr.fit(data[:,0:2], data[:,2] )

	print('Coefficients: \n', regr.coef_)
	print('Intercept:n', regr.intercept_)
	print('score:', regr.score(data[:,:2], data[:, 2]))











