"""
This is a scrip to implement PCA method using library
Implementor: Yihang Chen
n_components is the new k dimensions parameter

"""
#Group Member:
#Bin Zhang (5660329599)
#Yihang Chen (6338254416)
#Hanzhi Zhang (4395561906)

import numpy as np
from sklearn.decomposition import PCA 


def load_data(file_path):

	with open(file_path, 'r') as f:

		datas = ';'.join(f.readlines())
		data = np.mat(datas)
		# print(data)
		return data

	pass


if __name__ == '__main__':
	
	file_path = 'pca-data.txt'

	data = load_data(file_path)
	print(data)
	pca = PCA(n_components=2)

	reduced_data = pca.fit_transform(data)
	print(reduced_data)
