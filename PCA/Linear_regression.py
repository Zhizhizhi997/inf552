import numpy as np
import random
import matplotlib.pyplot as plt

#Group Member:
#Bin Zhang (5660329599)
#Yihang Chen (6338254416)
#Hanzhi Zhang (4395561906)

def load_dataset(txt_path,rate=0.99):
    with open(txt_path) as f:
        content = f.readlines()

    content = [row.replace('\n','') for row in content]
    Data = []
    for line in content:
        data = [float(i) for i in line.split(',')]
        Data.append(data)

    Data = np.array(Data) # (3000,3)
    num = Data.shape[0]
    
    index = int(rate*num)
    random.shuffle(Data)
    train_data = Data[0:index]
    test_data = Data[index:Data.shape[0]]

    
    train_set_x = train_data[:,0:-1].T #  (2700, 2)  we will use n*m format which means we need 2*2700
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


def propagate(w, b, X, Y):
    m = X.shape[1] # number of samples
    
    Y_hat = np.dot(w.T,X) + b
    
    cost = 1/(2*m) * np.sum(np.square(Y_hat-Y))     #   (1,m) 
    
    dw = 1/m*np.dot(X,(Y_hat-Y).T)  # (n,1) 
    db = 1/m * np.sum(Y_hat-Y)  
    
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

    

if __name__ == '__main__':

    txt_path = 'linear-regression.txt'
    train_set_x,train_set_y,test_set_x,test_set_y = load_dataset(txt_path)
    w,b = initilize_param(train_set_x.shape[0])
    params, grads, costs = optimize(w, b, train_set_x, train_set_y, num_iterations=10000, learning_rate=0.05)
    print(params['w'],params['b'])

