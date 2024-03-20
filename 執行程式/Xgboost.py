import xgboost as xgb
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import sklearn
import pandas as pd
import numpy as np
import pickle

def xg(stock,df):
    
    num = len(stock)-1
    mx = stock.iloc[:,1:14] #將要預測月營收的欄位與月營收切分
    mx = mx.drop([num],axis=0)
    minmax = preprocessing.MinMaxScaler()
    x = minmax.fit_transform(mx)
    y = stock.iloc[:,:1] #y為月營收歷史資料
    y = y.shift(-1)
    y = y.drop(num)
    y = y.values
#----------------------------------
    m1=[]
    for i in range(6): #分別取得各參數的值
        tar1 = df.iloc[:,i:i+1]
        tar1 = tar1.values
        tar1 = np.ravel(tar1)
        m1.append(tar1)
#----------------------------------
    x_train, x_test , y_train , y_test = train_test_split(x, y, test_size=0.1,random_state=42)
    xgbrModel = xgb.XGBRegressor(n_estimators = m1[0][0],learning_rate = m1[1][0],max_depth = m1[2][0],min_child_weight = m1[3][0],reg_lambda = m1[4][0],gamma = m1[5][0])
    xgbrModel.fit(x_train,y_train)
    y_pred = xgbrModel.predict(x_test)
    mape = sklearn.metrics.mean_absolute_percentage_error(y_test,y_pred)
    z = mape*100
    z = round(z,2)
    #print(z,"%")
#----------------------------------模型的預測(以上)，如要更換其他預測模型可直接抽換這部分
    pickle.dump(xgbrModel,open('C:/Users/User/Desktop/一條龍/模型放置區/2303',"wb")) #打包訓練好的模型，事實上可以不用多這步
    loaded_model = pickle.load(open('C:/Users/User/Desktop/一條龍/模型放置區/2303',"rb"))
    adc = stock.iloc[num:,1:14] #切分出最新月份的欄位
    pred = loaded_model.predict(adc)
    return pred #得到預測的下個月營收
    