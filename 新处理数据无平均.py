import numpy as np
import pandas as pd
import time
from sklearn import preprocessing
import math
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
aaa=['营业收入\n[报告期] 2013年报\n[报表类型] 合并报表\n[单位] 元',
'营业收入\n[报告期] 2014年报\n[报表类型] 合并报表\n[单位] 元',
'营业收入\n[报告期] 2015年报\n[报表类型] 合并报表\n[单位] 元',
'营业收入\n[报告期] 2016年报\n[报表类型] 合并报表\n[单位] 元',
'营业收入\n[报告期] 2017年报\n[报表类型] 合并报表\n[单位] 元',
'营业收入\n[报告期] 2018年报\n[报表类型] 合并报表\n[单位] 元']

from sklearn.preprocessing import MinMaxScaler,StandardScaler
Size = []
start = time.process_time()
for t in range(0, 5):
    print(t)
    X_train = []
    y_train = []
    X_eval = []
    y_eval = []
    X_com_eval = []
    y_com_eval = []
    X_test = []
    y_test = []
    X_com_test = []
    y_com_test = []


    for k in range(4, 10):
        df = pd.read_csv("stock_list.csv", encoding='utf-8-sig', index_col=0)
        print(k)
        df2 = df.sort_values(by=aaa[k - 4])
        df2=df2.iloc[int(t/5*len(df)):int((t+1)/5*len(df)),:]
        stock_code_list2 = df2.index.to_list()
        for n in range(0, len(stock_code_list2)):  # len(stock_code_list2)
            stock_code = stock_code_list2[n]
            data = pd.read_csv(
                'D:/PycharmProjects/QuantRL2/V4/V4/Data_pre/' + stock_code[0:6] +
                "_" + stock_code[7:9] + '.csv',
                encoding='utf-8-sig', index_col=0)
            data_real=np.array(data)
            data.fillna(method='bfill', inplace=True)
            data.fillna(method='ffill', inplace=True)
            data = data.fillna(0)

            data2 = data.iloc[:, 4].to_list()
            data3 = np.array(preprocessing.minmax_scale(
                data.iloc[0:11, 0:31], feature_range=(0, 1), axis=1, copy=True))[k - 4:k + 1, :]
            if k <= 7:
                X_train.append(data3)
                y_train.append(data.iloc[k + 1, 4] / 100)

            if k == 8:
                X_eval.append(data3)
                y_eval.append(data.iloc[k + 1, 4] / 100)



                if math.isnan(data_real[11, 31]) == False:
                    X_com_eval.append(data3)

                    y_com_eval.append(data.iloc[k + 1, 4] / 100)


            if k == 9:

                X_test.append(data3)
                y_test.append(data.iloc[k + 1, 4] / 100)


                if math.isnan(data_real[12, 31]) == False:
                    X_com_test.append(data3)
                    y_com_test.append(data.iloc[k + 1, 4] / 100)


    Size.append([np.array(X_train),
                 np.array(y_train).reshape(-1, 1),
                 np.array(X_eval),
                 np.array(y_eval).reshape(-1, 1),
                 np.array(X_test),
                 np.array(y_test).reshape(-1, 1),
                 np.array(X_com_eval),
                 np.array(y_com_eval).reshape(-1, 1),
                 np.array(X_com_test),
                 np.array(y_com_test).reshape(-1, 1),
                 ]) # di dao gao


Size = np.array(Size)

np.save(file="V3_data_noMA.npy", arr=Size)
end = time.process_time()
print('Running time —————— %s Seconds' % (end - start))

