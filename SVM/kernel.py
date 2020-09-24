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

    x,y,Q= [],[],[]
    for line in content:
        x.append([float(line[i]) for i in range(0,2)])
        y.append(int(line[-1]))

    x = np.array(x)                # x.shape (100,2)

    # kernal function  
    # (x1,x2) => (1, x1**2 , x2**2 , 2**0.5*x1 ,2**0.5*x2 , 2**0.5*x1*x2
    for xi in x: # increase the attribute dimension  
        Q.append([1,xi[0]**2,xi[1]**2,(2**0.5)*xi[0],(2**0.5)*xi[1],(2**0.5)*xi[0]*xi[1]]) 
    # here we specify the Q specifically which means we produce each z by hands
    # we can also use mapping to use kernel function automatically

    Q = np.matrix(Q)               # x.shape (100,6)
    y = np.array(y)                # y.shape  (100,)
    y = y.reshape(-1,1)            # (100,1)

    return x,y,Q



# solve OPP by cvxopt 
# use explicit way to sovle QPP ---> we derive z first 
def QPP(Q,y):
    ## objective function
    point_num = Q.shape[0]
    P_matrix = np.multiply(np.dot(Q,Q.T),np.dot(y,y.T)) 
    P_matrix = np.multiply(np.dot(Q,Q.T),np.dot(y,y.T))   # another   way to get the P

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



# kernel function  k(x,x^' )=〖(1+x^T x')〗^2
def kernel_func(x1,x2):
    # x1 x2 both are column vector  shape (d,1)
    return (1 + np.inner(x1, x2))**2

def mapping(x):
    # space dimension increase
    sample_size= x.shape[0]
    Q = np.zeros((sample_size, sample_size))
    # mapping
    for i, xi in enumerate(x):
        for j, xj in enumerate(x):
            Q[i, j] = kernel_func(xi, xj)
    
    return Q


# use kernel way to sovle QPP ---> we use mapping 
def QPP_kernel(x,y):
    ## objective function
    point_num = x.shape[0]
 
    Q = mapping(x) 
    P_matrix = np.multiply(np.dot(Q,Q.T),np.dot(y,y.T))   # another   way to get the P

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
def cal_w_b(sol,Q,y):
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
    w = sum(np.multiply(Q,lagrange_factors*y)).T


    # pick one support vector 
    y_k =y[point_index[0]]
    x_k =Q[point_index[0]].T
    b = y_k  -  np.dot(w.T,x_k)
    print('w:', w)
    print('b:', b)

    return w,b,point_index,lagrange_list

# list support vectors
def get_support_vector(x,sv_index):
    support_vector = []
    for index in sv_index:
        support_vector.append(x[index])

    print("support vectors:" ,support_vector)
    return support_vector


if __name__ == '__main__':
    x,y,Q= load_data('nonlinsep.txt')
    
    #explicit get z and calcute w,b 
    sol = QPP(Q,y)
    w,b,sv_index,lagrange_list = cal_w_b(sol,Q,y)
    get_support_vector(x,sv_index)


    #use kernel function automatically  and calcute w,b 
    sol2 = QPP_kernel(x,y)
    w2,b2,sv_index2,lagrange_list2 = cal_w_b(sol,Q,y)
    get_support_vector(x,sv_index2)



