import pandas as pd
import numpy as np
from keras import models, layers
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

def load_data(csv_file, is_death=False):
    df = pd.read_csv(csv_file)  # Afghanistan
    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y%m%d")
    df = df[['Date', 'TargetValue_x', 'TargetValue_y']]
    if is_death:
        df = df['TargetValue_y']
    else:
        df = df['TargetValue_x']
    return list(df)

def create_dataset(dataset, timestep=7):
    dataX, dataY = [], []
    if timestep == 7:  # use the 7 days data to predict 1 day later number
        for i in range(len(dataset) - timestep):
            a = dataset[i:(i + timestep)]
            dataX.append(a)
            dataY.append(dataset[i + timestep])

    if timestep == 15:  # use the 15 days data to predict 3 days later number
        for i in range(len(dataset) - timestep):
            sample_y = dataset[i + timestep: i + timestep + 3]
            if len(sample_y) == 3:
                sample_x = dataset[i: i + timestep]
                dataX.append(sample_x)  # index : index + day
                dataY.append(sample_y)

    if timestep == 30:  # use the 30 days data to predict the 7 days later number
        for i in range(len(dataset) - timestep):
            sample_y = dataset[i + timestep: i + timestep + 7]
            if len(sample_y) == 7:
                sample_x = dataset[i: i + timestep]
                dataX.append(sample_x)  # index : index + day
                dataY.append(sample_y)

    dataX, dataY = np.array(dataX), np.array(dataY)

    # seed = 7
    # np.random.seed(seed)
    # x_train, x_test, y_train, y_test = train_test_split(dataX,dataY , test_size=0.1, random_state=seed)
    # print('x_train.shape', x_train.shape)
    # print('y_train.shape', y_train.shape)
    # print('x_test.shape', x_test.shape)
    # print('y_test.shape', y_test.shape)

    # print(dataX.shape,dataY.shape)
    return dataX, dataY


def build_model(dataX, dayaY, epochs=100):
    def my_func(x):
        return abs(x)

    ##1.import data
    x_train, y_train = dataX, dayaY
    feature_num = x_train.shape[1]
    if y_train.ndim == 1:  # format  => (88,)
        output_num = 1
    else:
        output_num = y_train.shape[1]  # (79, 3) => 3

    if feature_num == 7:
        layers_shape = [8, 16]
    if feature_num == 15:
        layers_shape = [24, 16]
    if feature_num == 30:
        layers_shape = [48, 16]

    ##2、build a model
    model = models.Sequential()  # build a sequential model (MLP)
    model.add(layers.Dense(layers_shape[0], activation='relu', input_shape=(feature_num,)))
    model.add(layers.Dense(layers_shape[1], activation='relu'))
    model.add(layers.Dense(output_num))  # make predicts
    model.add(layers.Activation(my_func))  # abs(x)

    ##3、compile the model
    model.compile(optimizer='rmsprop',
                  loss='mse',
                  metrics=['mae'])

    ##4、train the model
    history = model.fit(x_train, y_train, batch_size=1, epochs=epochs, verbose=1)
    return model


def performace_test(model, dataX, dataY):
    ##5、make predictions on new data and make eavaluations
    pred_Y = model.predict(dataX)
    pred_r2 = r2_score(dataY, pred_Y)

    return pred_r2, pred_Y


def plot_data(dataY, pred_Y, country, pred_Y29=None):
    t1 = np.arange(1, len(dataY) + 1, 1)
    t2 = np.arange(len(dataY) - len(pred_Y) - len(pred_Y[0]) + 2 \
                   , len(dataY) + 1, 1)

    if pred_Y29 != None:
        t3 = np.arange(len(dataY) + 2, len(dataY) + 2 + len(pred_Y29))
    pred_Y = unpack_nest(pred_Y)

    plt.plot(t1, dataY, 's', label="confirmed case")  # truth data
    plt.plot(t2, pred_Y, 'r', label='predict curve')  # prediction for known data
    if pred_Y29 != None:
        t3 = np.arange(len(dataY) + 2, len(dataY) + 2 + len(pred_Y29))
        plt.plot(t3, pred_Y29, 's', label='predict case')
    plt.xlabel('days')
    plt.ylabel('number of cases')
    plt.suptitle('Cases in {}'.format(country), fontsize=16)
    plt.legend(loc=0)
    plt.show()


def unpack_nest(pred_Y):
    dict = {}
    for num in range(len(pred_Y)):
        for col in range(len(pred_Y[num])):
            date = num + col + 1
            dict[date] = dict.get(date, []) + [pred_Y[num][col]]
    res = []
    for num in range(len(dict)):
        res.append(sum(dict[num + 1]) / len(dict[num + 1]))
    return np.array(res)


def predict29(ds):
    ds = tuple(ds)
    train_a1 = list(ds)
    train_a2 = list(ds)
    train_a3 = list(ds)
    size = len(ds) - 1
    # print(size)
    day = size + 29 + 1
    # print(train_a1)
    for k in range(29):
        x1_test = []
        x2_test = []
        x3_test = []
        # print(x1_test)

        for v in range(6, -1, -1):
            size_use = size - v
            # print(train_a1[size_use])

            ip_v = train_a1[size_use]

            x1_test.append(ip_v)
            # print(x1_test)
            if v == 0:
                # print(x1_test)
                # print(x1_test)
                pred_y1 = model_week.predict(np.array([x1_test]))
                # print(pred_y1)
                train_a1.append(pred_y1.tolist()[0][0])

        if k % 3 == 0:
            for v in range(14, -1, -1):
                size_use = size - v

                ip_v = train_a2[size_use]

                x2_test.append(ip_v)
                if v == 0:
                    pred_y2 = model_halfmon.predict(np.array([x2_test]))
                    res2 = pred_y2.tolist()
                    for item in res2[0]:
                        train_a2.append(item)

        if k % 7 == 0:
            for v in range(29, -1, -1):
                size_use = size - v

                ip_v = train_a3[size_use]

                x3_test.append(ip_v)
                if v == 0:
                    pred_y3 = model_mon.predict(np.array([x3_test]))
                    res3 = pred_y3.tolist()
                    for issue in res3[0]:
                        train_a3.append(issue)

        size += 1
    size = len(ds)
    return train_a1[size:day], train_a2[size:day], train_a3[size:day]


def combine_predict(r2_week, r2_halfmon, r2_mon, pre_by_week, pre_by_halmon, pre_by_mon):
    if r2_week <= 0:
        r2_week = 0
    if r2_halfmon <= 0:
        r2_halfmon = 0
    if r2_mon <= 0:
        r2_mon = 0

    total_weight = r2_week + r2_halfmon + r2_mon
    week_weights = r2_week / total_weight
    halmon_weights = r2_halfmon / total_weight
    mon_weights = r2_mon / total_weight

    prediction_result = []
    for i in range(len(pre_by_week)):
        predict_data = week_weights * pre_by_week[i] + halmon_weights * pre_by_halmon[i] + mon_weights * pre_by_mon[i]
        print(predict_data)
        prediction_result.append(predict_data)

    return prediction_result


if __name__ == '__main__':

    print("Please input the country name (note:the name is the prefix csv in the data folder.)")
    country = input()
    csv_file = "../data/{}.csv".format(country)
    dataset = load_data(csv_file)

    # set timestep as 7
    time_step_week = 7
    x_week, y_week = create_dataset(dataset, timestep=time_step_week)
    model_week = build_model(x_week, y_week)  # model 1
    r2_week, pred_Y_week = performace_test(model_week, x_week, y_week)


    # set timestep as 15
    time_step_halfmon = 15
    x_halfmon, y_halfmon = create_dataset(dataset, timestep=time_step_halfmon)
    model_halfmon = build_model(x_halfmon, y_halfmon)  # model 2
    r2_halfmon, pred_Y_halfmon = performace_test(model_halfmon, x_halfmon, y_halfmon)

    # set timestep as 30
    time_step_mon = 30
    x_mon, y_mon = create_dataset(dataset, timestep=time_step_mon)
    model_mon = build_model(x_mon, y_mon)  # model 3
    r2_mon, pred_Y_mon = performace_test(model_mon, x_mon, y_mon)


    # PLOT THE PREICT DATA
    pre_by_week,pre_by_halmon, pre_by_mon = predict29(dataset)
    plot_data(np.array(dataset), pred_Y_week, country, pred_Y29=pre_by_week)
    plot_data(np.array(dataset), pred_Y_halfmon, country, pred_Y29=pre_by_halmon)
    plot_data(np.array(dataset), pred_Y_mon, country, pred_Y29=pre_by_mon)


    # get combine preditions
    final_predictions = combine_predict(r2_week, r2_halfmon, r2_mon, pre_by_week, pre_by_halmon, pre_by_mon)
    print(final_predictions)