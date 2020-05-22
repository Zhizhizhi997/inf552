from sklearn.cluster import KMeans
import numpy as np
from matplotlib import pyplot as plt

#Group Member:
#Bin Zhang (5660329599)
#Yihang Chen (6338254416)
#Hanzhi Zhang (4395561906)


def load_data(file_path):

	with open(file_path, 'r') as f:

		data = ';'.join(f.readlines())
		data = np.mat(data)
		# print(data)
	return data
	pass

def plot_result(data, labels, k):

	colors = ['r', 'b', 'g']

	data_lable = []
	for i in range(k):
		data_lable.append([])

	for i in range(len(data)):
		# print(data[i])
		data_lable[labels[i]].append([data[i][0,0], data[i][0,1]])


	for i in range(k):
		data_lable[i] = np.mat(data_lable[i])
		plt.plot(data_lable[i][:,0], data_lable[i][:,1], 'ob', color=colors[i])

	plt.show()

	pass




if __name__ == '__main__':
	
	k = 3

	file_path = 'clusters.txt'
	data = load_data(file_path)
	# print(data)
	kmeans = KMeans(n_clusters=k, random_state=0).fit(data)
	print(kmeans.labels_)

	plot_result(data, kmeans.labels_, k)


