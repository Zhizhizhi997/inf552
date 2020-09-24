import numpy as np
from cvxopt import solvers,matrix

#Group Member:
#Bin Zhang (5660329599)
#Yihang Chen (6338254416)
#Hanzhi Zhang (4395561906)


# import data and process it
def load_data(file_txt):
    with open(file_txt) as f:
        lines = f.readlines()
    content = [line.replace('\n','').split(',') for line in lines]

    x,y = [],[]
    for line in content:
        x.append([float(line[i]) for i in range(0,2)])
        y.append(int(line[-1]))

    x = np.matrix(x)                # x.shape (100,2)
    y = np.array(y)                 # y.shape  (100,)
    point_num = x.shape[0]

    y = y.reshape(-1,1)            # (100,1)
    x = np.matrix(x)               # x.shape (100,2)

    return x,y 


# solve QPP by cvxopt 
def QPP(x,y):
    ## objective function
    point_num = x.shape[0]
    P_matrix = np.multiply(np.dot(x,x.T),np.dot(y,y.T))
    P = matrix(P_matrix,tc="d")
    q = matrix(-np.ones((point_num,1)), tc="d")  #(100,1)

    # # inequality condition   
    G = matrix(-np.eye(point_num),tc="d") # (100,100)
    h = matrix(np.zeros((point_num,1)))   # (100,1)

    # equality condition
    A = matrix(y,(1,point_num),tc="d") #1*100 
    b = matrix(0.0) # (1,1)

    sol = solvers.qp(P,q,G,h,A,b)

    return sol

# get the hyperparameters
def cal_w_b(sol):
    lagrange_list = list(sol['x']) # get all lagrange factor
    support_rang_list = [] 
    point_index = []

    for i in range(0,len(lagrange_list)):
        if lagrange_list[i] >= 1e-06:
            support_rang_list.append(lagrange_list[i])
            point_index.append(i)
        else:
            support_rang_list.append(0)
     
        
    lagrange_factors = np.array(lagrange_list).reshape(-1,1) #(100,1)
    w = sum(np.multiply(x,lagrange_factors*y)).T

    # pick one support vector 
    y_k =y[point_index[0]]
    x_k =x[point_index[0]].T

    b = y_k  -  np.dot(w.T,x_k)

    print('w:', w)
    print('b:', b)

    return w,b,point_index


# list support vectors
def get_support_vector(x,sv_index):
    support_vector = []
    for index in sv_index:
        support_vector.append(x[index])

    print("support vectors:" ,support_vector)
    return support_vector


if __name__ == '__main__':
    x,y= load_data('linsep.txt')
    sol = QPP(x,y)
    w,b,point_index = cal_w_b(sol)
    get_support_vector(x,point_index)










