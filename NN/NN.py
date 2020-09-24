from PIL import Image
import numpy as  np

#Group Member:
#Bin Zhang (5660329599)
#Yihang Chen (6338254416)
#Hanzhi Zhang (4395561906)

## 1:import data and process it
def read_img(img_path):
    y = 1 if 'down' in img_path else 0   # get label y 
    img = Image.open(img_path)         # read image
    x = np.array(img).flatten().reshape(-1,1)
    
    return x,y

def read_imgs(data_txt):
    X,Y = [],[]  
    with open(data_txt) as f:
        lines = f.readlines()
        img_list = [line.replace('\n','') for line in lines]
    
    for img in img_list:
        x,y = read_img(img)
        X.append(x)
        Y.append(y)

    X = np.array(X)
    Y = np.array(Y)
        
    return X,Y

# ====》 X_train.shape # (184, 960, 1)   （sample_num, sample_attr_num,1） for sample xi its shape is (960, 1)
# ====》 Y_train.shape # (184,)          （sample_num,）                   for sample xi its label yi is a scalar

## 2. build a neural network
class NeuralNetwork():

    def __init__(self,X_train,X_test,Y_train,Y_test,epochs,learning_rate):
        self.X_train = X_train
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_test = Y_test 
        self.layer_dims = [self.X_train.shape[1],100,1] # network architecture
        self.L = len(self.layer_dims) #  number of layers in the network(including input layer)
        self.Layer_num = len(self.layer_dims) -1  #  number of layers in the network(without input layer)
        self.parameters = {}          #  parameters of the whole network
        self.epochs = epochs
        self.X = {} # record the each layer's output
        self.learning_rate = learning_rate

    def initialize_parameters(self):
        layer_dims = self.layer_dims
        for l in range(1,self.L):
            self.parameters['W' + str(l)] =     np.random.uniform(-0.01,0.01,size=[layer_dims[l-1],layer_dims[l]])
            self.parameters['b' + str(l)] =     np.random.uniform(-0.01,0.01,size=[layer_dims[l],1]) 
            self.parameters['Delta' + str(l)] = np.zeros((layer_dims[l],1)) 
            
    def linear_regression(self,x, w, b):  
        """
        in this function, we receive a (n,1) input and return a (m,1) output
        input:  last layer's output 
        output: this layer's temporary output(without sigmoid)
        """

        S = np.dot(w.T,x) + b  
        return S  # (100,1)

    def sigmoid(self,S):
        """
        return: the output of this layer
        """
        return 1/(1+np.exp(-S))

    def forward_propagation(self,x):
        """
        x  is a single sample of the dataset, it has d dimensions 
        parameters --   dictionary containing your parameters "W1", "b1","Delta1", "W2", "b2","Delta2"
                        W -- weight     (size of previous layer, size of  current layer)
                        b -- bias       (size of current layer,1)
                        Delta -- de/dS  (size of current layer,1)
        """
       
        L = self.Layer_num # layer numbers 2
        X = [x] # X record the ouput from each layer,  for the input layer, the output is x itself


        # calculate from 1 to L-1 layer
        for l in range(1,L):
            W = self.parameters["W" + str(l)]
            b = self.parameters["b" + str(l)]
            #linear  -> sigmoid  ->linear  -> sigmoid ->...
            S = self.linear_regression(X[l-1], W, b)  # X[l-1] from 0 
            A = self.sigmoid(S)          # get the output of this layer
        
            X.append(A)

        # calculate Lth layer : this is the last layer
        WL = self.parameters["W" + str(L)]
        bL = self.parameters["b" + str(L)]
        SL = self.linear_regression(X[-1], WL, bL)
        AL = self.sigmoid(SL) # here AL is the last result
        X.append(AL)

        for index,value in enumerate(X):
            self.X[index] = value


    def back_propagation(self,y):
        L = self.Layer_num
        result = self.X[L] # the result from the output layer
        #cost =  np.square(y - result)  standard squared error 

        base_case = 2*(result-y)*result*(1-result) # get the base case Dlata(l,1) 0.24565517   use mse as cost function
        # base_case =  result - y   # if use the log loss 
        
        self.parameters['Delta'+ str(L)]  = base_case # shape (1,1) only one number
        
        # update Delta for each layer
        for layer_level in range(1,L)[::-1]: # 1  from backwards to forwards
            # use vectorization
            Delta_last_l     = self.parameters['Delta' + str(layer_level)] 
            Delta_l = self.parameters['Delta' + str(layer_level+1)]
            W_l = self.parameters['W'+str(layer_level+1)]
            X_L = self.X[layer_level] 

            self.parameters['Delta' + str(layer_level)]  = np.dot(W_l,Delta_l)* X_L * (1-X_L) 
            
        # based on Delta, update b and W
        for l in range(1,self.L):
            BL = self.parameters['b' + str(l)]
            # use vectorization
            self.parameters['b' + str(l)] = BL - self.learning_rate*self.parameters['Delta' + str(l)]
        
        for l in range(1,self.L):
            WL = self.parameters['W' + str(l)]
            # use vectorization
            Delta_l = self.parameters['Delta' + str(l)]
            X_last_l = self.X[l-1] 

            self.parameters['W' + str(l)] = WL - self.learning_rate * np.dot(X_last_l,Delta_l.T)


    def train_model(self):
        print('Neural network is training now. It may take a while....')
        for i in range(self.epochs):
            for index,x in enumerate(self.X_train):
                y = self.Y_train[index] 
                self.forward_propagation(x)
                self.back_propagation(y)

    def predict(self):
        Y_pridict = []
        
        for index,x in enumerate(self.X_test): 
            self.forward_propagation(x)
            prob = self.X[self.Layer_num]  # get the final result
            if prob > 0.5:
                y_predict = 1
            else:
                y_predict = 0     

            Y_pridict.append(y_predict)

        Y_test = np.array(self.Y_test)
        Y_pridict = np.array(Y_pridict)
        accuracy = sum(Y_test==Y_pridict)/len(Y_pridict)
        print('Accuracy is '+str(accuracy)+" from " + str(self.epochs) +" epochs' training")


if __name__ == "__main__":
    ## get the data and transform it to a proper format
    X_train,Y_train =  read_imgs('downgesture_train.list')
    X_test,Y_test   =  read_imgs('downgesture_test.list')

    # build a nn     self,X_train,X_test,Y_train,Y_test,epochs,learning_rate
    epochs = 1000
    learning_rate = 0.1
    nn = NeuralNetwork(X_train,X_test,Y_train,Y_test,epochs,learning_rate)
    nn.initialize_parameters()
    nn.train_model()
    nn.predict()



