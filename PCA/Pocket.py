import numpy as np
import random

#Group Member:
#Bin Zhang (5660329599)
#Yihang Chen (6338254416)
#Hanzhi Zhang (4395561906)

file = open('classification.txt','r')
content = []
for line in file:
    line = line.strip('\n')
    line_list = line.split(',')
    #ignore the 5th line
    new_add = line_list[0:3]+[line_list[-1]]
    content.append(new_add)
    
#print(content)

# attrsact x
# initial weight

x_list = []
for line in content:
    a = ["-1"] + line[:-1]
    b= [eval(item) for item in a]
    x_list.append(b)
    
np_x = np.array(x_list)

num_shape = np.shape(np_x)
weight = np.zeros(num_shape[1])

def predict(weight, matrix_1,content_num):
    mul = np.dot(weight, matrix_1)
    if mul<0 and content_num[-1] == "+1":
        return False
    elif mul>=0 and content_num[-1] =='-1':
        return False
    else:
        return True
    
def train_for(np_x, weight, content, iteration):
    choice = []
    res =[]
    rate = 0.1
    for num in range(np.shape(np_x)[0]):
        res_one = predict(weight,np_x[num],content[num])
        res.append(res_one)
        if res_one == False:
            choice.append(num)
    if choice == []: # all data is satisfied
        return weight, 1
    
    choice_ran = random.sample(choice,1)
    weight2 = weight + eval(content[choice_ran[0]][-1])*rate*np_x[choice_ran[0]]
    accuracy = sum(res)/len(res)
    return accuracy, weight2

def pa(np_x, weight, content, accuracy = 0, result = None):
    iteration = 1
    while iteration < 7000:
        accuracy_new,weight_new = train_for(np_x,weight,content, iteration)
        if accuracy_new > accuracy:
            #print(accuracy_new, accuracy)
            #print(weight)
            weight = weight_new
            result = weight
            accuracy = accuracy_new
        else: # <
            weight = train_for(np_x,weight,content, iteration)[-1]
            pass
        
        iteration += 1
        
    return result,accuracy


b = pa(np_x, weight, content)
print(b)