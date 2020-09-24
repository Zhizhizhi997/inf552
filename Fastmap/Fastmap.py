import random
import copy
import numpy as np
import matplotlib.pyplot as plt


#Group Member:
#Bin Zhang (5660329599)
#Yihang Chen (6338254416)
#Hanzhi Zhang (4395561906)

def load_data(txt_path):
    with open(txt_path) as f:
        content = [row.replace('\n', '') for row in f.readlines()]

    if content[0].isalpha(): # load the word list
        word_dict = {}
        for index,word in enumerate(content):
            word_dict[index+1] = word
        return word_dict

    else: # load the distance
        data_list = []
        for row in content:
            numbers = row.split('\t')
            numbers = [ int(i) for i in numbers] 
            data_list.append(numbers)
    
        return data_list


def pick_pivot(word_dict,data_list,DISTANCE_ITERATIONS=5):
    '''
    find the points a and b that the distance between them are farest 
    '''
    a,b = random.sample(word_dict.keys(),2)
    for i in range(DISTANCE_ITERATIONS):
        c = farest_point(a,data_list)
        if c == b: #  a<->b is farest
            break
        b = c
        d = farest_point(b,data_list)
        if d == a: #  a<->b‘ is farest
            break
        a = d
        
    return a, b


def farest_point(point_a,data_list):
    '''
    get the farest point from point a  , return the point
    '''

    distance_info = []
    for i in data_list:
        if i[0] == point_a or i[1] == point_a:
            distance_info.append(i)
            
    # get the largest number meaning the farest distance 
    max_distance = max([i[-1] for i in distance_info ]) 
 
    # get the index of these points 
    for i in distance_info:
        if i[-1] == max_distance:
            pivot1,pivot2 =i[0],i[1] # if multiple points have the farest distance, we use the first one 

            if point_a == pivot1:
                return pivot2
            else:
                return pivot1


data_path = 'fastmap-data.txt'
word_path = 'fastmap-wordlist.txt'
word_dict  = load_data(word_path)
data_list = load_data(data_path)

def get_dist(o1,o2,data_list):
    if o1 == o2:
        return 0 
    for row in data_list:
        if int(row[0]) == o1 and int(row[1]) == o2 or int(row[0]) == o2 and int(row[1]) == o1:
            return row[-1]
        
def compute_xi(i,oa,ob,data_list):
    Oai = get_dist(i,oa,data_list)
    Oab = get_dist(oa,ob,data_list)
    Obi = get_dist(i,ob,data_list)
    return (Oai**2+Oab**2-Obi**2)/(2*Oab)
        
    
def update_dist(j,X,data_list):
    '''
    recursive conditions
    '''

    for row in data_list:
        xi = X[row[0]][j]
        xj = X[row[1]][j]
        row[-1] = np.sqrt(row[-1]**2 - (xi-xj)**2)
    return 


def fastmap(k,data_list,word_dict):
    
    oa,ob = pick_pivot(word_dict,data_list)
    
    X = {}
    for j in range(k): # recursively run k times 

        for i in word_dict.keys():
            xi = compute_xi(i,oa,ob,data_list)
            X[i] = X.get(i,[])+[xi]


        update_dist(j,X,data_list)
        oa,ob = pick_pivot(word_dict,data_list)
    
    return X



if __name__ == '__main__':
    data_path = 'fastmap-data.txt'
    word_path = 'fastmap-wordlist.txt'
    word_dict  = load_data(word_path)
    data_list = load_data(data_path)

    new_data = fastmap(2,data_list,word_dict)

    X = []
    Y = []
    for key,value in new_data.items():
        X.append(value[0])
        Y.append(value[1])

    # print the result
    oupt_dict={}
    for index,value in enumerate(list(word_dict.values())):
        oupt_dict[value] = new_data[index+1]
    print(oupt_dict)

    # Plot the points
    fig, ax = plt.subplots()
    for i in list(word_dict.keys()):
        ax.scatter(X[i-1], Y[i-1])
        ax.annotate(word_dict[i], (X[i-1], Y[i-1]))
    plt.show()



