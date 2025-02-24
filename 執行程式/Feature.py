import pandas as pd
import numpy as np

def fe(num,month,lt,x):#index(月份) month(本月營收) lt(股權數)
    rows = len(x)
    lym = x.loc[rows-12,'單月營收(千元)'] #去年月營收
    lmm = x.loc[rows-1,'單月營收(千元)'] #上月營收
    
    累計營收 = 0
    去年累計營收 = 0
    近3月累計營收 = 0
    近12月累計營收 = 0
    去年近3月 = 0
    去年近12月 = 0
    for i in range(num-1): #一些需要加總計算的欄位
        m1 = x.loc[rows-1-i,'單月營收(千元)']
        累計營收 = 累計營收 + m1
        m2 = x.loc[rows-1-i,'去年單月營收(千元)']
        去年累計營收 = 去年累計營收 + m2
    
    for i in range(2):
        m1 = x.loc[rows-1-i,'單月營收(千元)']
        近3月累計營收 = 近3月累計營收 + m1
        m2 = x.loc[rows-1-i,'去年單月營收(千元)']
        去年近3月 = 去年近3月 + m2
        
    for i in range(11):  
        m1 = x.loc[rows-1-i,'單月營收(千元)']
        近12月累計營收 = 近12月累計營收 + m1
        m2 = x.loc[rows-1-i,'去年單月營收(千元)']
        去年近12月 = 去年近12月 + m2
    
    近12月累計營收 = 近12月累計營收 + month #套入計算公式，算出相關欄位
    近3月累計營收 = 近3月累計營收 + month
    累計營收 = 累計營收 + month
    去年近3月 = 去年近3月 + lym
    去年近12月 = 去年近12月 + lym
    去年累計營收 = 去年累計營收 + lym
    單月營收成長率 = ((month/lym)-1)*100
    單月營收與上月比 = ((month-lmm)/lmm)*100
    累計營收成長率 = ((累計營收/去年累計營收)-1)*100
    近3月累計營收成長率 = ((近3月累計營收-去年近3月)/abs(去年近3月))*100
    近12月累計營收成長率 = ((近12月累計營收-去年近12月)/abs(去年近12月))*100
    單月每股營收 = month/lt
    近3月每股營收 = 近3月累計營收/lt
    近12月每股營收 = 近12月累計營收/lt
    累計每股營收 = 累計營收/lt
    
    累計營收成長率 = np.round(累計營收成長率,2)
    單月營收成長率 = np.round(單月營收成長率,2)
    單月營收與上月比 = np.round(單月營收與上月比,2)
    近3月累計營收 = np.round(近3月累計營收,2)
    近12月累計營收 = np.round(近12月累計營收,2)
    近3月累計營收成長率 = np.round(近3月累計營收成長率,2)
    近12月累計營收成長率 = np.round(近12月累計營收成長率,2)
    單月每股營收 = np.round(單月每股營收,2)
    近3月每股營收 = np.round(近3月每股營收,2)
    近12月每股營收 = np.round(近12月每股營收,2)
    累計每股營收 = np.round(累計每股營收,2)
    
    new = pd.DataFrame({'單月營收(千元)':month,
                      '去年單月營收(千元)':lym,
                      '單月營收成長率％':單月營收成長率,
                      '單月營收與上月比％':單月營收與上月比,
                      '累計營收成長率％':累計營收成長率,
                      '近 3月累計營收(千元)':近3月累計營收,
                      '近12月累計營收(千元)':近12月累計營收,
                      '近12月累計營收成長率':近12月累計營收成長率,
                      '近3月累計營收成長率':近3月累計營收成長率,
                      '單月每股營收(元)':單月每股營收,
                      '近12月每股營收':近12月每股營收,
                      '近 3月每股營收':近3月每股營收,
                      '累計每股營收(元)':累計每股營收
                     },index=[rows])
    
    x = pd.concat([x,new],axis=0)
    return x #最後輸出為一個dataframe