"""
feed forward NN completed by library sklearn
ouput 10 rounds individual training and testing

Network structure:
	input layer
	full connect layer 100
	output layer 1

Using SGD solver
Learning rate: 0.1

coded by Yihang Chen
"""

#Group Member:
#Bin Zhang (5660329599)
#Yihang Chen (6338254416)
#Hanzhi Zhang (4395561906)

from PIL import Image
import numpy as np
from sklearn.neural_network import MLPClassifier
import random 


class BP_FFNN:

	def __init__(self):

		self.params = {'solver':'sgd', 'learning_rate': 'constant', 'learning_rate_init':0.01, 'activation':'logistic',
						'max_iter':1000, 'hidden_layer_sizes':(100,)}


		self.mlp = MLPClassifier(**self.params)


		pass

	def read_PGM(self, filepath):

		img = Image.open(filepath)
		return list(img.getdata())

	def load_data_set(self, file_list):

		with open(file_list, 'r') as f:
			data = f.read().splitlines()

		data_list  = [d for d in data]
		
		x_data = []
		y_data = []
		for t in data_list:
			y_data.append(1 if 'down' in t else 0)
			x_data.append(self.read_PGM(t))
		return x_data, y_data


	def run(self):

		train_x_data, train_y_data = self.load_data_set('downgesture_train.list')
		test_x_data, test_y_data = self.load_data_set('downgesture_test.list')

		for i in range(3):
			self.mlp.fit(train_x_data, train_y_data)
			print('round{}'.format(i+1))
			print('train scores:', self.mlp.score(train_x_data, train_y_data))
			print('test scores:', self.mlp.score(test_x_data, test_y_data))
			print('parameters:', self.mlp.get_params())
			print('')



if __name__ == '__main__':

	NN = BP_FFNN()
	
	NN.run()
