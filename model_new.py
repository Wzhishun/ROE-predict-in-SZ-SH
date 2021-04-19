from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import tensorflow as tf
pd.set_option('display.max_columns', 9000)
pd.set_option('display.max_rows', 9000)
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import backend as K
a=[]
b=[]
epochs = 1000
eval_true = []
eval_pre = []
test_true = []
test_pre = []
eval_ana_true = []
eval_ana_pre = []
test_ana_true = []
test_ana_pre = []
def r_square(y_true, y_pred):
    SS_res = K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )
def layer_with_ma(inputs):
    split_feature1, split_feature2, split_feature3, \
    split_feature4, split_feature5, split_feature6, \
    split_feature7, split_feature8, split_feature9, split_feature10 = \
        tf.split(inputs, [8, 2, 3, 3, 2, 1, 6, 4, 2, 1], 2)
    a1, a2, a3, a4, a5 = tf.split(split_feature10, [1, 1, 1, 1, 1], 1)
    split_feature11 = tf.concat([a1,a2,a3],axis=2)
    YINGLI = split_feature1
    ZIBEN = tf.concat([split_feature3, split_feature6], axis=2)
    CHANHGZHAIYINGYUN = tf.concat([split_feature4, split_feature8], axis=2)
    CHENGZHANG = split_feature7
    QITA = tf.concat([split_feature2, split_feature5, split_feature9], axis=2)
    tf1 = tf.keras.layers.GRU(32, activation=tf.nn.relu)(YINGLI)
    tf1 = tf.keras.layers.Dropout(0.25)(tf1)
    tf1 = tf.keras.layers.Dense(32, activation=tf.nn.relu)(tf1)
    tf1 = tf.keras.layers.Dense(8, activation=tf.nn.relu)(tf1)

    tf2 = tf.keras.layers.GRU(32, activation=tf.nn.relu)(ZIBEN)
    tf2 = tf.keras.layers.Dropout(0.25)(tf2)
    tf2 = tf.keras.layers.Dense(32, activation=tf.nn.relu)(tf2)
    tf2 = tf.keras.layers.Dense(4, activation=tf.nn.relu)(tf2)

    tf3 = tf.keras.layers.GRU(32, activation=tf.nn.relu)(CHANHGZHAIYINGYUN)
    tf3 = tf.keras.layers.Dropout(0.25)(tf3)
    tf3 = tf.keras.layers.Dense(32, activation=tf.nn.relu)(tf3)
    tf3 = tf.keras.layers.Dense(7, activation=tf.nn.relu)(tf3)

    tf4 = tf.keras.layers.GRU(32, activation=tf.nn.relu)(CHENGZHANG)
    tf4 = tf.keras.layers.Dropout(0.25)(tf4)
    tf4 = tf.keras.layers.Dense(32, activation=tf.nn.relu)(tf4)
    tf4 = tf.keras.layers.Dense(6, activation=tf.nn.relu)(tf4)

    tf5 = tf.keras.layers.GRU(32, activation=tf.nn.relu)(QITA)
    tf5 = tf.keras.layers.Dropout(0.25)(tf5)
    tf5 = tf.keras.layers.Dense(32, activation=tf.nn.relu)(tf5)
    tf5 = tf.keras.layers.Dense(6, activation=tf.nn.relu)(tf5)

    tf_la = tf.concat([tf1, tf2, tf3, tf4, tf5], axis=1)
    #tf_la = tf.keras.layers.Dense(16, activation=tf.nn.relu, name="tf_la_dense")(tf_la)
    tf_la = tf.keras.layers.Flatten()(tf_la)
    outputs = tf.keras.layers.Dense(1, activation=tf.nn.relu)(tf_la)
    split_feature11 =tf.keras.layers.Flatten()(split_feature11)
    tf_lb= tf.concat([outputs,split_feature11],axis=1)
    outputs = tf.keras.layers.Dense(32, activation=tf.nn.relu)(tf_lb)
    outputs = tf.keras.layers.Dense(16, activation=tf.nn.relu)(outputs)
    outputs = tf.keras.layers.Dense(16, activation=tf.nn.relu)(outputs)
    outputs = tf.keras.layers.Dense(1)(outputs)
    return outputs

def layer_with_ma2(inputs): # final05
    split_feature1, split_feature2, split_feature3, \
    split_feature4, split_feature5, split_feature6, \
    split_feature7, split_feature8, split_feature9, split_feature10 = \
        tf.split(inputs, [8, 2, 3, 3, 2, 1, 6, 4, 2, 1], 2)
    a1, a2, a3, a4, a5 = tf.split(split_feature10, [1, 1, 1, 1, 1], 1)
    split_feature11 = tf.concat([a1, a2, a3], axis=2)
    YINGLI = split_feature1
    ZIBEN = tf.concat([split_feature3, split_feature6], axis=2)
    CHANHGZHAIYINGYUN = tf.concat([split_feature4, split_feature8], axis=2)
    CHENGZHANG = split_feature7
    QITA = tf.concat([split_feature2, split_feature5, split_feature9], axis=2)


    tf1 = tf.keras.layers.Conv1D(64,kernel_size=1,strides=2)(YINGLI)
    tf1 = tf.keras.layers.Dropout(0.25)(tf1)
    tf1 = tf.keras.layers.GRU(16, activation=tf.nn.relu)(tf1)
    tf1 = tf.keras.layers.Dense(8, activation=tf.nn.relu)(tf1)

    tf2 = tf.keras.layers.Conv1D(64, kernel_size=1,strides=2)(ZIBEN)
    tf2 = tf.keras.layers.Dropout(0.25)(tf2)
    tf2 = tf.keras.layers.GRU(16, activation=tf.nn.relu)(tf2)
    tf2 = tf.keras.layers.Dense(8, activation=tf.nn.relu)(tf2)

    tf3 = tf.keras.layers.Conv1D(64, kernel_size=1,strides=2)(CHANHGZHAIYINGYUN)
    tf3 = tf.keras.layers.Dropout(0.25)(tf3)
    tf3 = tf.keras.layers.GRU(16, activation=tf.nn.relu)(tf3)
    tf3 = tf.keras.layers.Dense(8, activation=tf.nn.relu)(tf3)

    tf4 = tf.keras.layers.Conv1D(64, kernel_size=1,strides=2)(CHENGZHANG)
    tf4 = tf.keras.layers.Dropout(0.25)(tf4)
    tf4 = tf.keras.layers.GRU(16, activation=tf.nn.relu)(tf4)
    tf4 = tf.keras.layers.Dense(8, activation=tf.nn.relu)(tf4)

    tf5 = tf.keras.layers.Conv1D(64, kernel_size=1,strides=2)(QITA)
    tf5 = tf.keras.layers.Dropout(0.25)(tf5)
    tf5 = tf.keras.layers.GRU(16, activation=tf.nn.relu)(tf5)
    tf5 = tf.keras.layers.Dense(8, activation=tf.nn.relu)(tf5)

    tf_la = tf.concat([tf1, tf2, tf3, tf4, tf5], axis=1)
    tf_la = tf.keras.layers.Flatten()(tf_la)

    tf_la = tf.keras.layers.Dropout(0.25)(tf_la)
    outputs = tf.keras.layers.Dense(1, activation=tf.nn.relu)(tf_la)
    split_feature11 =tf.keras.layers.Flatten()(split_feature11)
    tf_lb= tf.concat([outputs,split_feature11],axis=1)

    outputs = tf.keras.layers.Dense(32, activation=tf.nn.relu)(tf_lb)

    outputs = tf.keras.layers.Dense(32, activation=tf.nn.relu)(outputs)
    outputs = tf.keras.layers.Dense(16, activation=tf.nn.relu)(outputs)
    outputs = tf.keras.layers.Dense(1)(outputs)


    '''
    tf_la = tf.concat([tf1, tf2, tf3, tf4, tf5], axis=1)
    #tf_la = tf.keras.layers.Dense(16, activation=tf.nn.relu, name="tf_la_dense")(tf_la)
    tf_la = tf.keras.layers.Flatten()(tf_la)
    outputs = tf.keras.layers.Dense(1, activation=tf.nn.relu)(tf_la)
    split_feature11 =tf.keras.layers.Flatten()(split_feature11)
    tf_lb= tf.concat([outputs,split_feature11],axis=1)
    outputs = tf.keras.layers.Dense(32, activation=tf.nn.relu)(tf_lb)
    outputs = tf.keras.layers.Dense(16, activation=tf.nn.relu)(outputs)
    outputs = tf.keras.layers.Dense(16, activation=tf.nn.relu)(outputs)
    outputs = tf.keras.layers.Dense(1)(outputs)
    '''
    return outputs




data=np.load(file="V3_data.npy",allow_pickle=True)
Size=[]

for m in range(1,2):
    '''
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
                 np.array(sma_train).reshape(-1, 1)
                 ]) # di dao gao
    '''
    X_train = data[4-m][0]
    y_train = data[4-m][1]
    X_eval = data[4-m][2]
    y_eval = data[4-m][3]
    X_test = data[4-m][4]
    y_test = data[4-m][5]
    X_eval2 = data[4-m][6]
    y_eval2 = data[4-m][7]
    X_test2 = data[4-m][8]
    y_test2 = data[4-m][9]
    y_traina = data[4-m][10]
    last_train = data[4-m][11]
    ma_train = data[4-m][12]
    sma_train =data[4-m][13]
    y_eval_real= data[4-m][14]
    y_test_real= data[4-m][15]
    y_eval3= data[4-m][16]
    y_test3= data[4-m][17]



    mm = MinMaxScaler()

    '''for t in range(len(y_train)):
        if  y_train[4-t]<-0.2:
            y_train[4-t]=-0.2
        if  ma_train[4-t]<-0.2:
            ma_train[4-t]=-0.2
        if  sma_train[4-t]<-0.2:
            sma_train[4-t]=-0.2
        if  last_train[4-t]<-0.2:
            last_train[4-t]=-0.2'''

    y = mm.fit_transform(np.concatenate([y_train, y_eval],axis=0).reshape(-1,1))
    y_train_mimmax = y[0:len(y_train)]
    y_eval_minmax = y[len(y_train):len(y_train)+len(y_test)]
    print(y_train.tolist())
    print(mm.inverse_transform(y_train_mimmax).tolist())


    inputs = tf.keras.Input(shape=(5, 32,))
    outputs = layer_with_ma2(inputs=inputs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    # print(model.summary())
    model.compile(loss='mse',optimizer='Adam',metrics=[r_square])
    model.fit(X_train, y_train_mimmax,
              batch_size=1024, epochs=epochs,
              validation_data=(X_eval, y_eval_minmax),)
    y_predict2 = model.predict(x=X_eval, batch_size=1024, verbose=0)
    y_predict2 = mm.inverse_transform(y_predict2)

    mse = mean_squared_error(y_eval, y_predict2)
    r2 = r2_score(y_eval, y_predict2)

    eval_true=eval_true+list(y_eval)
    eval_pre = eval_pre + list(y_predict2)



    a.append(round(mse, 4))
    b.append(round(r2, 4))
    # _____________________________________________________
    y_predict_2 = model.predict(x=X_eval2, batch_size=1024, verbose=0)
    y_predict_2 = mm.inverse_transform(y_predict_2)
    print(y_predict_2)
    mse = mean_squared_error(y_eval3, y_predict_2)
    r2 = r2_score(y_eval3, y_predict_2)
    eval_ana_true = eval_ana_true + list(y_eval2)
    eval_ana_pre = eval_ana_pre + list(y_predict_2)
    a.append(round(mse, 4))
    b.append(round(r2, 4))
    # _____________________________________________________
    y_predict = model.predict(x=X_test, batch_size=1024, verbose=0)
    y_predict = mm.inverse_transform(y_predict)
    mse = mean_squared_error(y_test, y_predict)
    r2 = r2_score(y_test, y_predict)

    test_true = test_true + list(y_test)
    test_pre = test_pre + list(y_predict)


    a.append(round(mse, 4))
    b.append(round(r2, 4))
    # _____________________________________________________
    y_predict_1 = model.predict(x=X_test2, batch_size=1024, verbose=0)
    y_predict_1 = mm.inverse_transform(y_predict_1)
    print(y_predict_1)
    mse = mean_squared_error(y_test3, y_predict_1)
    r2 = r2_score(y_test3, y_predict_1)
    test_ana_true = test_ana_true + list(y_test2)
    test_ana_pre = test_ana_pre + list(y_predict_1)
    a.append(round(mse, 4))
    b.append(round(r2, 4))
    # _____________________________________________________
for i in range(0,len(a)):
    print(a[i])
for i in range(0,len(b)):
    print(b[i])

mse = mean_squared_error(eval_true, eval_pre)
r2 = r2_score(eval_true, eval_pre)
print(mse)
print(r2)
mse = mean_squared_error(eval_ana_true, eval_ana_pre)
r2 = r2_score(eval_ana_true, eval_ana_pre)
print(mse)
print(r2)

mse = mean_squared_error(test_true, test_pre)
r2 = r2_score(test_true, test_pre)
print(mse)
print(r2)
mse = mean_squared_error(test_ana_true, test_ana_pre)
r2 = r2_score(test_ana_true, test_ana_pre)
print(mse)
print(r2)
new = pd.concat([pd.DataFrame(eval_true),pd.DataFrame(eval_pre),pd.DataFrame(eval_ana_true),pd.DataFrame(eval_ana_pre),
                 pd.DataFrame(test_true),pd.DataFrame(test_pre),pd.DataFrame(test_ana_true),pd.DataFrame(test_ana_pre)],axis=1)
new.columns=["2018real","2018pre","2018ana_real","2018ana_pre","2019real","2019pre","2019ana_real","2019ana_pre"]
new.to_csv("final02.csv")
