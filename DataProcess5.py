import numpy as np
import pandas as pd
import time
from sklearn import preprocessing
import math
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
a=['营业收入\n[报告期] 2013年报\n[报表类型] 合并报表\n[单位] 元',
'营业收入\n[报告期] 2014年报\n[报表类型] 合并报表\n[单位] 元',
'营业收入\n[报告期] 2015年报\n[报表类型] 合并报表\n[单位] 元',
'营业收入\n[报告期] 2016年报\n[报表类型] 合并报表\n[单位] 元',
'营业收入\n[报告期] 2017年报\n[报表类型] 合并报表\n[单位] 元',
'营业收入\n[报告期] 2018年报\n[报表类型] 合并报表\n[单位] 元']
df = pd.read_csv("stock_list.csv", encoding='utf-8-sig', index_col=0)
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
    for k in range(4, 10):
        print(k)
        df2 = df.sort_values(by=a[k - 4])
        df2.to_csv("sort1.csv")
        df2=df2.iloc[int(t/5*len(df)):int((t+1)/5*len(df)),:]
        stock_code_list2 = df2.index.to_list()

        for i in range(0, len(stock_code_list2)):  # len(stock_code_list2)
            print(i)
            stock_code = stock_code_list2[i]
            data = pd.read_csv(
                'D:/PycharmProjects/QuantRL2/V4/V4/Data_pre/' + stock_code[0:6] +
                "_" + stock_code[7:9] + '.csv',
                encoding='utf-8-sig', index_col=0)
            data_real=np.array(data)

            for m in range(0, 11):
                for n in range(0, 31):
                    data.iloc[m, n] = np.float64(str(data.iloc[m, n]).replace(',', ''))
            data.fillna(method='bfill', inplace=True)
            data.fillna(method='ffill', inplace=True)
            data = data.fillna(0)

            data2 = np.array(preprocessing.minmax_scale(
                data.iloc[0:11,0:31], feature_range=(0, 1), axis=1, copy=True))[k - 4:k + 1, :]
            if k <= 7:
                X_train.append(data2)

                y_train.append(data.iloc[k + 1, 4] / 100)
            if k == 8:
                X_eval.append(data2)
                y_eval.append(data.iloc[k + 1, 4] / 100)
                if math.isnan(data_real[11, 31]) == False:
                    X_com_eval.append(data2)

                    y_com_eval.append(data.iloc[k + 1, 4] / 100)
                    y_ana_eval.append(data_real[11, 31] / 100)
            if k == 9:
                X_test.append(data2)
                y_test.append(data.iloc[k + 1, 4] / 100)
                if math.isnan(data_real[12, 31]) == False:
                    X_com_test.append(data2)
                    y_com_test.append(data.iloc[k + 1, 4] / 100)
                    y_ana_test.append(data_real[12, 31] / 100)
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
                 np.array(y_ana_eval).reshape(-1, 1),
                 np.array(y_ana_test).reshape(-1, 1),

                 ])
Size = np.array(Size)
print(Size.shape, Size[0][0].shape,Size[0][3].shape)
np.save(file="V3_data_5.npy", arr=Size)
end = time.process_time()
print('Running time —————— %s Seconds' % (end - start))

