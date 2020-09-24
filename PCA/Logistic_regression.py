import numpy as np
import random
import matplotlib.pyplot as plt

#Group Member:
#Bin Zhang (5660329599)
#Yihang Chen (6338254416)
#Hanzhi Zhang (4395561906)

def load_dataset(txt_path,rate=1):
    with open(txt_path) as f:
        content = f.readlines()
        content = [line.replace('\n','') for line in content]

    Data = []
    for line in content:
        data = [float(i) for i in line.split(',')]
        data.pop(-2) # remove the second last item
        if data[-1] ==-1:
            data[-1] = 0
        Data.append(data)

    Data = np.array(Data) # (3000,3)
    num = Data.shape[0]

    index = int(rate*num)
    random.shuffle(Data)
    train_data = Data[0:index]
    test_data = Data[index:Data.shape[0]]

    train_set_x = train_data[:,0:-1].T 
    train_set_y = train_data[:,-1].reshape(1,-1)
    test_set_x = test_data[:,0:-1].T    
    test_set_y = test_data[:,-1].reshape(1,-1)

    print ("train_set_x shape: " + str(train_set_x.shape))
    print ("train_set_y shape: " + str(train_set_y.shape))
    print ("test_set_x shape: " + str(test_set_x.shape))
    print ("test_set_y shape: " + str(test_set_y.shape))

    return train_set_x,train_set_y,test_set_x,test_set_y

def initilize_param(feature_num):
    w = np.zeros((feature_num, 1))
    b = 0
    return w, b


def sigmoid(z):
    return 1/(1+np.exp(-z))
    
def propagate(w, b, X, Y):
    m = X.shape[1] # number of samples
    Z = np.dot(w.T,X) + b    # (n,m)
    A = sigmoid(Z)           # (1,m)
    cost = -1/m * np.sum(Y * np.log(A) + (1-Y) * np.log(1-A))     #   (1,m)  
    dw = 1/m*np.dot(X,(A-Y).T)  # (n,1) 
    db = 1/m * np.sum(A-Y)  
    
    grads = {"dw": dw,
             "db": db}
    
    return grads,cost



def optimize(w, b, X, Y, num_iterations, learning_rate):
    
    costs = []
    for i in range(num_iterations):
        
        grads,cost = propagate(w,b,X,Y)
        dw = grads["dw"]
        db = grads["db"]
        w = w - learning_rate * dw
        b = b - learning_rate * db
        costs.append(cost)
    
    params = {"w": w,
              "b": b}

    grads = {"dw": dw,
             "db": db}
        
    return params, grads, costs


def y_hat_generated(w,b,X):

    m = X.shape[1]
    Y_hat = np.zeros((1,m))
    w = w.reshape(X.shape[0], 1)
    A = sigmoid(np.dot(w.T, X) + b)
    Y_hat = np.rint(A)
     
    return Y_hat

# merge all the function and bulid a logistc regresiion model
def model(X_train, Y_train, X_test, Y_test, num_iterations = 7000, learning_rate = 0.05):

    w, b = initilize_param(X_train.shape[0])

    # Gradient descent 
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate)
    
    # Retrieve parameters w and b from dictionary "parameters"
    w = parameters["w"]
    b = parameters["b"]
    
    # Predict test/train set examples 
    Y_test_train = y_hat_generated(w, b, X_train)
    Y_hat_test = y_hat_generated(w, b, X_test)
    
    print("accuracy: {} %".format(100 - np.mean(np.abs(Y_test_train - Y_train)) * 100))
    print(w,b)
    
    return 


if __name__ == '__main__':
    txt_path = 'classification.txt'
    train_set_x,train_set_y,test_set_x,test_set_y = load_dataset(txt_path) # atucally here I set all data as training set 
    # test 
    # w,b = initilize_param(train_set_x.shape[0])
    # params, grads, costs = optimize(w, b, train_set_x, train_set_y, num_iterations=7000, learning_rate=0.05)
    # print(params['w'],params['b'])

    model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=7000, learning_rate=0.05)
