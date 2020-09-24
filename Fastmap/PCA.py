import numpy as np

#Group Member:
#Bin Zhang (5660329599)
#Yihang Chen (6338254416)
#Hanzhi Zhang (4395561906)

def process_data(txt_path):
    '''
    load data and process data
    '''
    with open(txt_path) as f:
        content = f.readlines()
    data = []
    for row in content:
        row =[float(i) for i in  row.replace('\n','').split('\t')]
        data.append(row)
    
    data = np.array(data)
    # normalization 
    mean = data.mean(axis=0) # get each colume'e mean
    data = data - mean # x- u for each colume 
    
    return data
    

def compute_cov(X):    
    '''
    Compute the covariance matrix
    '''
    num= X.shape[0]
    return 1/num * np.dot(X.T,X)


def eigenvector_value(Cov):
    '''
    Compute eigenvectors and eigenvalue
    '''
    eigenvector ,eigenvalue = np.linalg.eig(Cov) 
    return eigenvector,eigenvalue


def feature_vector(eigenvalue,eigenvector,k):
    '''
    Pick top k principal components
    '''
    
    assert k <= eigenvector.shape[0]
    
    eign_value = sorted(list(eigenvalue),reverse=True)
    index_list = []
    for i in range(k):
        value = eign_value[i]
        index = np.argwhere(eigenvalue == value)  
        index_list.append(int(index))
        
    select_vector = eigenvector.take(index_list,axis = 1)
    return select_vector


def generate_data(X,reduced_matrix):
    '''
    Deriving the new data set
    '''
    
    return np.dot(X,reduced_matrix)
    


if __name__ == '__main__':
    data = process_data('pca-data.txt')   
    cov = compute_cov(data)
    eigenvalue , eigenvector = eigenvector_value(cov)
    reduced_matrix = feature_vector(eigenvalue,eigenvector,2)   
    new_data = generate_data(data,reduced_matrix)
    print(new_data)
    print(new_data.shape)
    

