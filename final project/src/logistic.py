import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def load_data(csv_file, sequence_len=7, is_death=False):
    df = pd.read_csv(csv_file)  # Afghanistan
    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y%m%d")
    df = df[['Date', 'TargetValue_x', 'TargetValue_y']]

    # get the confirmed case number or death number
    if is_death:
        location = 2
    else:
        location = 1

    time_step, case_add, case_total = [], [], []
    sum_cases = 0
    for index, row in df.iterrows():
        time_step.append(index + 1)
        sum_cases += df.iloc[index][location]
        case_total.append(sum_cases)  # location=1 => confirmed case   location=2 => death
        case_add.append(df.iloc[index][location])

    # now we have to cut down some useless data( case number is 0 for long time ) ==> find the initial time of case
    time_length = len(time_step)

    t = np.array(time_step)  # time series [1,2,3,4,5.....]
    P = np.array(case_total)  # case series [20,30,40,50,60.........]
    return t, P

def logistic_increase_function(t,K,P0,r): # increase number
    # t:time   t0:initial time    P0:initial_value    K:capacity  r:increase_rate
    t0=1
    r=0.18
    exp_value=np.exp(r*t)
    return (K*exp_value*P0)/(K+(exp_value-1)*P0)

def plot_data(t,P,P_predict,future,future_predict,country):
    # plot
    plot1 = plt.plot(t, P, 's', label="confirmed case") #  truth data
    plot2 = plt.plot(future, future_predict, 's', label='predict cases')  # predict  for unknow data
    plot3 = plt.plot(t, P_predict, 'r', label='predict curve')            #  prediction for known data
    plt.xlabel('days')
    plt.ylabel('number of confirmed cases')
    plt.suptitle('Cases in {}'.format(country),fontsize=16)
    plt.legend(loc=0)
    plt.show()

def get_day_case(updated_case,future_predict):
    daily_case = []
    first_pre_case = future_predict[0]-updated_case
    daily_case.append(
first_pre_case)
    for i in range(1,len(future_predict)):
        case = future_predict[i]-future_predict[i-1]
        daily_case.append(case)

    return daily_case
if __name__ == '__main__':
    print("Please input the country name (note:the name is the prefix csv in the data folder.)")
    country = input()
    t, P = load_data('../data/{}.csv'.format(country), is_death=False)

    # use lse to fit the curve
    # popt, pcov = curve_fit(logistic_increase_function, t, P)
    popt, pcov = curve_fit(logistic_increase_function, t, P)
    K = popt[0]
    P0 = popt[1]
    r = popt[2]
    print("K", popt[0], "P0:", popt[1], 'r:', popt[2])

    # predictions by the curve for the known days
    P_predict = logistic_increase_function(t, popt[0], popt[1], popt[2])

    # predictions by the curve for the unknown days
    future = np.arange(t[-1]+1,t[-1]+1+29,1)
    future_predict = logistic_increase_function(future, popt[0], popt[1], popt[2])

    # get the daily prediction
    updated_case = P[-1]
    daily_case = get_day_case(updated_case,future_predict)
    print(daily_case)

    # plot
    plot_data(t, P,P_predict , future, future_predict,country)