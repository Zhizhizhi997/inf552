import numpy as np 
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

'''
using Polynomial regression to fit data. 
here we set an array of continuous 'offset' days as train tuple and predict next 'test_num' days data
as tried, we set degree of Polynomial computing 2 to get better result

'''

if __name__ == '__main__':
    
    print("Please input the country name (note:the name is the prefix csv in the data folder.)")
    country = input()

    train = pd.read_csv('../data/{}.csv'.format(country))
    # build the model by sklearn
    model = Pipeline([('poly', PolynomialFeatures(degree=2)),('linear', LinearRegression(fit_intercept=False))]) 

    test_num = 3
    offset = 3

    # train[:-10].values.shape
    y = train['TargetValue_x'].values

    x = np.array([y[i:i+offset] for i in range(y.shape[0]-offset)])
    y = np.array([y[i+offset] for i in range(y.shape[0]-offset)])

    x_train = x[0:-test_num]
    y_train = y[0:-test_num]

    x_test = x[-test_num:]
    y_test = y[-test_num:]

    x_total = x
    y_total = y

    model.fit(x_train,y_train)
    model.score(x_train,y_train)

    plt.plot(np.arange(y_total.shape[0]), y_total, 's', label='confrimed cases')
    plt.plot(np.arange(y_test.shape[0])+x_total.shape[0] - offset, model.predict(x_test), 's', label='predict cases')

    plt.plot(np.arange(x_total.shape[0]), model.predict(x_total), 'r', label='regression curve')
    plt.xlabel('days')
    plt.ylabel('number of confirmed cases')
    plt.suptitle('Cases in {}'.format(country),fontsize=16)
    plt.legend(loc=0)
    plt.show()


    print(model.score(x_train,y_train))
    print(model.score(x_test, y_test))
    print(model.predict(x_test))
    print(y_test)


    ## prediction the next 29
    total = list(x_total[-1])
    for i in range(29):
        # get the lastest 3 days number
        latest = np.array(total[i:i + 3]).reshape(1, -1)
        y_pre = model.predict(latest)
        if y_pre <= 0:
            y_pre = 0
        total.append(float(y_pre))

    predictions_num = list(total[3:])

    print("preditions", predictions_num)