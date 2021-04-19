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
    y_ana_eval = []
    X_test = []
    y_test = []
    X_com_test = []
    y_com_test = []
    y_ana_test = []
    ma_train = []
    sma_train = []
    last_train = []

    ma_eval = []
    sma_eval = []
    last_eval = []
    new1_eval = []
    new2_eval = []
    ma_ana_eval = []
    sma_ana_eval = []
    last_ana_eval = []
    new1_ana_eval = []
    new2_ana_eval = []

    ma_test = []
    sma_test = []
    last_test = []
    new1_test = []
    new2_test = []
    ma_ana_test = []
    sma_ana_test = []
    last_ana_test = []
    new1_ana_test = []
    new2_ana_test = []


    y_eval_real = []
    y_test_real = []
    y_eval3 = []
    y_test3 = []
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
                last_train.append(data2[k] / 100)
                ma_train.append(sum(data2[k - 4:k + 1]) / 500)
                sma_train.append(data2[k - 4] / 1500 + data2[k - 3] * 2 / 1500 + data2[k - 2] * 3 / 1500
                           + data2[k - 1] * 4 / 1500 + data2[k] * 5 / 1500)

            if k == 8:
                X_eval.append(data3)
                y_eval.append(data.iloc[k + 1, 4] / 100)


                y_eval_real.append(data.iloc[k + 1, 4] / 100)


                last_eval.append(data2[k] / 100)
                ma_eval.append(sum(data2[k - 4:k + 1]) / 500)
                sma_eval.append(data2[k - 4] / 1500 + data2[k - 3] * 2 / 1500 + data2[k - 2] * 3 / 1500
                                + data2[k - 1] * 4 / 1500 + data2[k] * 5 / 1500)
                if math.isnan(data_real[11, 31]) == False:
                    X_com_eval.append(data3)

                    y_com_eval.append(data.iloc[k + 1, 4] / 100)

                    y_eval3.append(data.iloc[k + 1, 4] / 100)

                    y_ana_eval.append(data_real[11, 31] / 100)
                    last_ana_eval.append(data2[k] / 100)
                    ma_ana_eval.append(sum(data2[k - 4:k + 1]) / 500)
                    sma_ana_eval.append(data2[k - 4] / 1500 + data2[k - 3] * 2 / 1500 + data2[k - 2] * 3 / 1500
                                        + data2[k - 1] * 4 / 1500 + data2[k] * 5 / 1500)
            if k == 9:

                X_test.append(data3)
                y_test.append(data.iloc[k + 1, 4] / 100)

                y_test_real.append(data.iloc[k + 1, 4] / 100)


                last_test.append(data2[k] / 100)
                ma_test.append(sum(data2[k - 4:k + 1]) / 500)
                sma_test.append(data2[k - 4] / 1500 + data2[k - 3] * 2 / 1500 + data2[k - 2] * 3 / 1500
                                + data2[k - 1] * 4 / 1500 + data2[k] * 5 / 1500)
                if math.isnan(data_real[12, 31]) == False:
                    X_com_test.append(data3)
                    y_com_test.append(data.iloc[k + 1, 4] / 100)

                    y_test3.append(data.iloc[k + 1, 4] / 100)


                    y_ana_test.append(data_real[12, 31] / 100)
                    last_ana_test.append(data2[k] / 100)
                    ma_ana_test.append(sum(data2[k - 4:k + 1]) / 500)
                    sma_ana_test.append(data2[k - 4] / 1500 + data2[k - 3] * 2 / 1500 + data2[k - 2] * 3 / 1500
                                        + data2[k - 1] * 4 / 1500 + data2[k] * 5 / 1500)

    y_train_a = y_train
    mm = MinMaxScaler()

    y = mm.fit_transform(np.array(y_train+ma_train+sma_train+last_train).reshape(-1,1))
    ma_train = y[len(y_train):len(y_train) + len(ma_train)]
    sma_train = y[len(y_train) + len(ma_train):len(y_train) + len(ma_train) + len(sma_train)]
    last_train = y[len(y_train) + len(ma_train) + len(sma_train):]
    new = np.zeros((len(y_train), 2))
    new = np.concatenate([ma_train, sma_train, last_train, new], axis=1)
    X_train2 = []
    for i in range(0, len(X_train)):
        a = np.concatenate([X_train[i], new[i, :].reshape(5, 1)], axis=1)
        X_train2.append(a)

    mm2 = MinMaxScaler()
    y2 = mm2.fit_transform(np.array(y_eval+ma_eval+sma_eval+last_eval).reshape(-1,1))
    ma_eval = y2[len(y_eval):len(y_eval) + len(ma_eval)]
    sma_eval = y2[len(y_eval) + len(ma_eval):len(y_eval) + len(ma_eval) + len(sma_eval)]
    last_eval = y2[len(y_eval) + len(ma_eval) + len(sma_eval):]
    new = np.zeros((len(y_eval), 2))
    print(len(y_eval), len(ma_eval), len(sma_eval), len(last_eval))
    new = np.concatenate([ma_eval, sma_eval, last_eval, new], axis=1)
    X_eval2 = []
    for i in range(0, len(X_eval)):
        a = np.concatenate([X_eval[i], new[i, :].reshape(5, 1)], axis=1)
        X_eval2.append(a)


    mm3 = MinMaxScaler()
    y = mm3.fit_transform(np.array(y_test+ ma_test+ sma_test+ last_test).reshape(-1,1))
    ma_test = y[len(y_test):len(y_test) + len(ma_test)]
    sma_test = y[len(y_test) + len(ma_test):len(y_test) + len(ma_test) + len(sma_test)]
    last_test = y[len(y_test) + len(ma_test) + len(sma_test):]
    new = np.zeros((len(y_test), 2))
    new = np.concatenate([ma_test, sma_test, last_test, new], axis=1)
    X_test2 = []
    for i in range(0, len(X_test)):
        a = np.concatenate([X_test[i], new[i, :].reshape(5, 1)], axis=1)
        X_test2.append(a)

    mm4 = MinMaxScaler()
    y = mm4.fit_transform(np.array(y_com_eval+ma_ana_eval+sma_ana_eval+last_ana_eval).reshape(-1,1))

    ma_ana_eval = y[len(y_com_eval):len(y_com_eval) + len(ma_ana_eval)]
    sma_ana_eval = y[len(y_com_eval) + len(ma_ana_eval):len(y_com_eval) + len(ma_ana_eval) + len(sma_ana_eval)]
    last_ana_eval = y[len(y_com_eval) + len(ma_ana_eval) + len(sma_ana_eval):]
    print(len(y_com_eval), len(ma_ana_eval), len(sma_ana_eval), len(last_ana_eval))
    new = np.zeros((len(y_com_eval), 2))
    new = np.concatenate([ma_ana_eval, sma_ana_eval, last_ana_eval, new], axis=1)
    X_ana_eval2 = []
    for i in range(0, len(X_com_eval)):
        a = np.concatenate([X_com_eval[i], new[i, :].reshape(5, 1)], axis=1)
        X_ana_eval2.append(a)

    mm5 = MinMaxScaler()
    y = mm5.fit_transform(np.array(y_com_test+ ma_ana_test+ sma_ana_test+last_ana_test).reshape(-1,1))

    ma_ana_test = y[len(y_com_test):len(y_com_test) + len(ma_ana_test)]
    sma_ana_test = y[len(y_com_test) + len(ma_ana_test):len(y_com_test) + len(ma_ana_test) + len(sma_ana_test)]
    last_ana_test = y[len(y_com_test) + len(ma_ana_test) + len(sma_ana_test):]
    new = np.zeros((len(y_com_test), 2))



    new = np.concatenate([ma_ana_test, sma_ana_test, last_ana_test, new], axis=1)
    X_ana_test2 = []
    for i in range(0, len(X_com_test)):
        a = np.concatenate([X_com_test[i], new[i, :].reshape(5, 1)], axis=1)
        X_ana_test2.append(a)

    Size.append([np.array(X_train2),
                 np.array(y_train).reshape(-1, 1),
                 np.array(X_eval2),
                 np.array(y_eval).reshape(-1, 1),
                 np.array(X_test2),
                 np.array(y_test).reshape(-1, 1),
                 np.array(X_ana_eval2),
                 np.array(y_com_eval).reshape(-1, 1),
                 np.array(X_ana_test2),
                 np.array(y_com_test).reshape(-1, 1),
                 np.array(y_train_a).reshape(-1, 1),
                 np.array(last_train).reshape(-1, 1),
                 np.array(ma_train).reshape(-1, 1),
                 np.array(sma_train).reshape(-1, 1),
                 np.array(y_eval_real).reshape(-1, 1),
                 np.array(y_test_real).reshape(-1, 1),
                 np.array(y_eval3).reshape(-1, 1),
                 np.array(y_test3).reshape(-1, 1)


                 ]) # di dao gao


Size = np.array(Size)

np.save(file="V3_data.npy", arr=Size)
end = time.process_time()
print('Running time —————— %s Seconds' % (end - start))

