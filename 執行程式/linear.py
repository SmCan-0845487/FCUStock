import requests
import json
import pandas as pd
import numpy as np
import sklearn
from sklearn import preprocessing
from sklearn import linear_model #為了跑 Linear中的lasso
from sklearn.model_selection import train_test_split

def lin(stock,add,stock_num): #分別輸入對應股票代號 季營收 股權數，以此獲取推測的現金股利
    headers = {
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/101.0.4951.67 Safari/537.36'
    }
    r = requests.get('https://statementdog.com/api/v2/fundamentals/'+stock+'/2018/2022/cf?qbu=false&qf=analysis',headers = headers)
    r.json()
    stock_json = r.json()
    stock_df = pd.DataFrame.from_dict(stock_json['quarterly']['NetIncomeMargin'])
    Stock_df = stock_df.iloc[:, 1:2]
    Stock_df.columns = ['稅後淨利率']
    float_lst = []
    a = Stock_df['稅後淨利率'][0:19] #19是要隨季數增長而改的
    for i in range(0,19):
        x = float(a.values[i][1])
        float_lst.append(x)
    a_df = pd.DataFrame(float_lst)  
    a_df.columns = ['稅後淨利率']
#----------------------------------------------
    stock_df = pd.DataFrame.from_dict(stock_json['quarterly']['Revenue'])
    Stock_df = stock_df.iloc[:, 1:2]
    Stock_df.columns = ['營收']
    float_lst2 = []
    b = Stock_df['營收'][0:19] #19是要隨季數增長而改的
    for i in range(0,19):
        x = float(b.values[i][1])
        float_lst2.append(x)
    b_df = pd.DataFrame(float_lst2)
    b_df.columns = ['營收']
#----------------------------------------------
    g = requests.get('https://statementdog.com/api/v2/fundamentals/'+stock+'/2021/2022/cf?qbu=false&qf=analysis',headers = headers)
    g.json()
    stock_json2 = g.json()
    stock_df = pd.DataFrame.from_dict(stock_json2['quarterly']['EPS'])
    Stock_df = stock_df.iloc[:, 1:2]
    Stock_df.columns = ['單季EPS']
    float_lst4 = []
    d = Stock_df['單季EPS'][0:7]
    for i in range(0,7):
        x = float(d.values[i][1])
        float_lst4.append(x)
    d_df = pd.DataFrame(float_lst4)
    d_df.columns = ['單季EPS']
#----------------------------------------------
    n = requests.get('https://statementdog.com/api/v2/fundamentals/'+stock+'/2017/2021/cf?qbu=false&qf=analysis',headers = headers)
    n.json()
    stock_json1 = n.json()
    stock_df = pd.DataFrame.from_dict(stock_json1['yearly']['DividendPayoutRatio'])
    Stock_df = stock_df.iloc[:, 1:2]
    Stock_df.columns = ['現金股利發放率']
    float_lst3 = []
    c = Stock_df['現金股利發放率']
    for i in range(0,5): #5是要隨年數增長而改的
        x=float(c.values[i][1])
        float_lst3.append(x)
    c_df = pd.DataFrame(float_lst3)
    c_df.columns = ['現金股利發放率']
    a = c_df.loc[0,'現金股利發放率']
    for i in range(len(c_df)-1):
        x = c_df.loc[i+1,'現金股利發放率']
        a+=x
    配息率 = a/500 #因為是使用5年平均，抓取5年除500，依此增加減少
#---------------------------------------------- 配息率 = (現金股利/eps)*100%
    mq1=[]#裝各季稅後淨利率 mq1為第一季類推
    mq2=[]
    mq3=[]
    mq4=[]
    msq1=[]#裝各季營收
    msq2=[]
    msq3=[]
    msq4=[]
    for i in range(19):
        if i%4==0:
            tar1 = a_df[i:i+1]#某年的第一季稅後淨利率
            tar2 = b_df[i:i+1]#某年的第一季季營收
            tar1 = tar1.values
            tar2=tar2.values
            mq1.append(tar1)
            msq1.append(tar2)
        if i%4==1:
            tar1 = a_df[i:i+1]
            tar2 = b_df[i:i+1]
            tar1=tar1.values
            tar2=tar2.values
            mq2.append(tar1)
            msq2.append(tar2)
        if i%4==2:
            tar1 = a_df[i:i+1]
            tar2 = b_df[i:i+1]
            tar1=tar1.values
            tar2=tar2.values
            mq3.append(tar1)
            msq3.append(tar2)
        if i%4==3:
            tar1 = a_df[i:i+1]
            tar2 = b_df[i:i+1]
            tar1=tar1.values
            tar2=tar2.values
            mq4.append(tar1)
            msq4.append(tar2)
    mq1 = np.ravel(mq1) #降維
    mq2 = np.ravel(mq2)
    mq3 = np.ravel(mq3)
    mq4 = np.ravel(mq4)
    msq1 = np.ravel(msq1)
    msq2 = np.ravel(msq2)
    msq3 = np.ravel(msq3)
    msq4 = np.ravel(msq4)
#---------------------------------------------------- lasso預測季稅後淨利率，原有新增多項次處理，發現多支股票默認效果較好，就無添加   
    df1 = pd.DataFrame (mq4, columns = ['稅後淨利率'])
    df2 = pd.DataFrame (msq4, columns = ['營收'])    
    minmax = preprocessing.MinMaxScaler()
    df3 = minmax.fit_transform(df2)
    X_train, X_test , Y_train , Y_test = train_test_split(df3,df1, test_size = 0.2, random_state = 42)
    reg = linear_model.Lasso()
    model = reg.fit(X_train,Y_train)
    new_df = pd.concat([df2,add],axis=0)
    new_df = minmax.fit_transform(new_df) #將正規化的數值還原
    正規化四 = new_df[4]
    t4 = []
    t4.append(正規化四)
    t4 = np.atleast_2d(t4)
    Y4 = model.predict(t4)
    Y4 = np.ravel(Y4)
    Y4 = np.round(Y4,2) #得到預測季營收所對應的季稅後淨利率       
#--------------------------------------------
    季度量 = len(add)
    YEPS = 0
    ans = 0
    x1 = add[0:1].values
    a = Y4/100
    a = a*x1
    a = a/stock_num
    a = np.round(a,2)#利用公式計算得出推測的
    YEPS+=a 
#--------------------------------------------        
    for i in range(4-季度量):
        b = len(d_df)-3+季度量+i
        x = d_df[b-1:b].values
        print(x)
        YEPS+=x
    print(a)
    年EPS = YEPS[0][0]
    年EPS = np.round(年EPS,2)#利用季eps加總得到年eps
    print("-------------------------------")
    print("推估年EPS:",年EPS)
    print("平均配息率:",np.round(配息率*100,2),"%")
    ans = 年EPS*配息率 #利用年eps與現金股利發放率計算得到推測的現金股利
    ans = np.round(ans,2)
    return ans

        