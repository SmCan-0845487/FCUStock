import numpy as np
import pandas as pd
from pandas import DataFrame
import Feature # 利用預測的月營收 產生對應feature的副程式
import Xgboost # 可以更換使用的模型 此次使用xgboost模型
import linear # 利用預測的季營收推出現金股利
df = pd.read_csv('C:/Users/User/Desktop/一條龍/全50.csv') 
df2 = pd.read_csv('C:/Users/User/Desktop/一條龍/股權數.csv')
df6 = pd.read_csv('C:/Users/User/Desktop/一條龍/參數.csv')
#df4 = pd.read_csv('C:/Users/User/Desktop/輸出.csv') #輸出結果之用

ran = 106 #106要隨著月數增加而增加，目前到2014/1月到2022/10月是106

def company_split(company):#傳入公司代碼(string)
    code1 = np.array(df.iloc[0:,0:1])
    for i in range(0,49): #從49之股票抓取匹配的股票代號(台灣50中刪除了1隻)
        tar1=code1[i*ran:i*ran+1]
        if(company==str(tar1[0])[2:6]):
            for j in range(i*ran,i*ran+1):
                a = df[j:j+ran] #股票對應之歷史欄位
    code2 = np.array(df2.iloc[0:,0:1])
    for i in range(0,49):
        tar2=code2[i:i+1]
        if(company==str(tar2[0])[2:6]):
            b = df2[i:i+1] #對應之股權數
    return a,b
data = {'n_estimators':[100], #xgboost模型的默認參數，如未找到對應的參數會用此替代
        'learning_rate':[0.3],
        'max_depth':[6],
        'min_child_weight':[1],
        'reg_lambda':[1],
        'gamma':[0]}
df8 = DataFrame(data)
def company_para(company): #抓取股票對應的參數
    z = df8
    code1 = np.array(df6.iloc[0:,0:1])
    for i in range(len(df6)):
        tar1=code1[i:i+1]
        if(company==str(tar1[0])[1:5]):
            z = df6[i:i+1]
            z = z.iloc[:,1:]
    return z

company = '2883' #輸入股票代號
catch,lt = company_split(company)
catch = catch.reset_index(drop=True)
df7 = company_para(company)
df7 = df7.reset_index(drop=True)
stock = catch.iloc[:,2:15] #將預測所不會使用到的股票代號及年月欄位切掉，以下也做相關處理
lt = lt.iloc[:,1:2]
lt = lt.values
lt = int(lt[0][0])
#--------------------------------------------------------------資料獲取(以上)
num = len(stock)%12
month = Xgboost.xg(stock,df7) #輸入歷史資料與參數，預測出月營收
month = int(month)
a = Feature.fe(num+1,month,lt,stock) #輸入順序(資料的月份 預測的月營收 股票股權數)，並輸出添加過新欄位的歷史資料a
print(num+1,"月","月營收",month)
for i in range(11-num): #重複做到今年12月，也可以藉由更改數值11推出更後面的月營收預測，但準度會越來越不准
    month = Xgboost.xg(a,df7)
    month = int(month)
    a = Feature.fe(num+1,month,lt,a)
    月數 = num+i+2
    if 月數 > 12:
        月數=月數-12
    print(月數,"月","月營收",month)
print("月報結束")
#--------------------------------------------------------------月營收預測(以上)
float_lst = []
length = int((12-num)/3)
for i in range(length+1): #利用預測的月營收加總，計算出四季營收
    季營收 = 0
    for j in range(3):
        季營收+=a.loc[len(a)+j-(length+1)*3+i*3,'單月營收(千元)']
    float_lst.append(季營收)
df3 = pd.DataFrame(float_lst, columns = ['營收']) #新增季營收的dataframe
現金股利 = linear.lin(company,df3,lt) #輸入對應股票代號 季營收 股權數，以此獲取推測的現金股利
print("殖利率 7% 的股價:",np.round(現金股利/0.07,2))
print("殖利率 8% 的股價:",np.round(現金股利/0.08,2))
print("殖利率 9% 的股價:",np.round(現金股利/0.09,2))
output = {'七':[np.round(現金股利/0.07,2)],
         '八':[np.round(現金股利/0.08,2)],
         '九':[np.round(現金股利/0.09,2)]}
'''df5 = DataFrame(output) #輸出csv檔之用
df4 = df4.append(df5,ignore_index=False)
df4.to_csv("C:/Users/User/Desktop/輸出.csv",encoding='utf_8_sig')'''