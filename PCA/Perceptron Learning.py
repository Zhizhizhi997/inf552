import numpy as np

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
    new_add = line_list[0:-1]
    content.append(new_add)


# attrsact x
# initial weight

x_list = []
for line in content:
    a = ["1"] + line[:-1]
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


def train(np_x,weight,content):
    res = [False]
    n =1 # iteration limitation
    while sum(res) < len(res):
        res = []
        rate = 1/n
        for num in range(np.shape(np_x)[0]):
            res_one = predict(weight,np_x[num],content[num])
            if res_one == False:
                weight = weight + eval(content[num][-1])*rate*np_x[num]
            res.append(res_one)
            #print(weight)
        n += 1
        #print(weight)
        if n > 7000:
            print('iteration is up  to limitation')
            break
    return weight, sum(res)/len(res)




weight = train(np_x,weight,content)

print(weight)