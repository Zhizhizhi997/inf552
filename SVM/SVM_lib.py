'''
implementation of SVM algo to tow dataset
sklearn support tow kind of linear SVM, basically, the LinearSVC is support
for larger dataset

In the mathmatical method used in library, it has upper limitation
of alpha, we set C to float('inf') so that we can nearly assume that alpha 
does not has the upper limit, so the result function is closer to the result
of our hand-write code.

the kernel used for nonlinear dataset is rbf

coded by Yihang Chen
'''

#Group Member:
#Bin Zhang (5660329599)
#Yihang Chen (6338254416)
#Hanzhi Zhang (4395561906)


import numpy as np
from sklearn.svm import SVC
from sklearn.svm import LinearSVC


def load_data(filepath):

	data = np.loadtxt(filepath, delimiter = ',', dtype=float)
	return data

def SVM_withKernel(x, y, kernel):

	clf = SVC(kernel = kernel, gamma = 'auto', C = float('inf'))
	clf.fit(x, y)
	if kernel == 'linear':
		print('W:', clf.coef_)
		pass

	print('b:', clf.intercept_)
	print('SV:', clf.n_support_)
	print('score:', clf.score(x,y))


def SVM_LinearSVC(x,y):
	clf = LinearSVC()
	clf.fit(x, y)
	print('score:', clf.score(x,y))



if __name__ == '__main__':
	
	print('*'*50)
	print('linear dataset')
	data = load_data('linsep.txt')
	x = data[:,0:2]
	y = data[:,2]
	SVM_withKernel(x,y, 'linear')
	SVM_LinearSVC(x,y)

	print('*'*50)
	print('nonlinear dataset')
	data = load_data('nonlinsep.txt')
	x = data[:,0:2]
	y = data[:,2]
	SVM_withKernel(x,y, 'rbf')
	# SVM_LinearSVC(x,y)

