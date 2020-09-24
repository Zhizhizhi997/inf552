import pandas as pd

train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')
sample = pd.read_csv('./data/submission.csv')

sub_train_cc = train[train['Target'] == 'ConfirmedCases']
# sub_train_cc
sub_train_fa = train[train['Target'] == 'Fatalities']
res = sub_train_cc.merge(sub_train_fa, how = 'left', on=['County', 'Province_State', 'Country_Region', 'Date'])
res = res[['County', 'Province_State', 'Country_Region', 'Date', 'Target_x', 'TargetValue_x', 'Target_y', 'TargetValue_y']]

sub_res = res.drop_duplicates(subset = ['Country_Region'], keep = 'first', inplace=False)


for i in sub_res['Country_Region'].values:
    print(i)
    res_sub = res[res['Country_Region'] == i]
    res_sub[res_sub['Province_State'].isna()].to_csv('./processed_data/sum_province/{}.csv'.format(i.replace('*', '')))
