from PIL import Image
import numpy as  np
from keras.models import Sequential 
from keras.layers import Dense
from keras import optimizers
from keras import models
from keras import layers

#Group Member:
#Bin Zhang (5660329599)
#Yihang Chen (6338254416)
#Hanzhi Zhang (4395561906)

## 1:import data and process it
def read_img(img_path):
    y = 1 if 'down' in img_path else 0   # get label y 
    img = Image.open(img_path)         # read image
    x = np.array(img).flatten()
    
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

# ====》 X_train.shape # (184, 960)      （sample_num, sample_attr_num） for sample xi its shape is (960, 1)
# ====》 Y_train.shape # (184,)          （sample_num,）                   for sample xi its label yi is a scalar
X_train,Y_train =  read_imgs('downgesture_train.list.txt')
X_test,Y_test   =  read_imgs('downgesture_test.list.txt')


# use keras to build a neural network
network = models.Sequential()
network.add(layers.Dense(100, activation='sigmoid', input_shape=(960,))) 
network.add(layers.Dense(1, activation='sigmoid'))

sgd = optimizers.SGD(lr=0.1)
network.compile(optimizer=sgd, loss='mse',metrics=['accuracy'])

network.fit(X_train, Y_train, epochs=1000, batch_size=1)

test_loss, test_acc = network.evaluate(X_test, Y_test)
print('test_acc:', test_acc)