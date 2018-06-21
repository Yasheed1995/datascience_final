import pandas as pd
import numpy as np
#效用函數 U = Expected_Return - 0.5*k*variance    (k = 風險趨避指數)  (我用Ln_Weight代表"長期獲利期望值"的權重，(1-Ln_Weight)是"短期獲利期望值"的權重)

Ln_Weight = 0.5
k = 0.5

df1 = pd.read_csv('stock.txt')
for i in range (0,len(df1)):
    df = pd.read_csv(str(df1.ix[i,'Name'])+'.TW.csv') #利用stock.txt上的Name欄位中的股票代號，讀取相對應的.csv
    print(df1.ix[i,'Name'])

    #求出變異數&長期的投資報酬率
    for j in range(0,len(df)):
        df.at[j,'Ln_Return'] = (df.loc[j,'Adj Close']/df.loc[1,'Adj Close']-1) #將每日的收盤價除以第一天的收盤價，作為報酬率/使用Adj 才能將除權除息的股票價值偏差計入
    vari = np.var(df['Ln_Return'],ddof=1) #對Return 取變異數 (ddof=1) 代表使用分母為N-1的sampling variance
    print('Variance = ')
    print(vari)
    df1.at[i,'Variance'] = vari #寫入stock.txt
    exp_return = np.mean(df['Ln_Return']) #np.mean 求Return的平均值
    print('Ln_Expected_Return = ')
    print(exp_return)
    df1.at[i,'Ln_Expected_Return'] = exp_return #寫入stock.txt

    #求出變異數短期的投資報酬率
    for j in range(len(df)-31,len(df)):
        df.at[j,'Sh_Return'] = (df.loc[j,'Adj Close']/df.loc[len(df)-31,'Adj Close']-1) #將每日的收盤價除以一個月前的收盤價，作為報酬率
    exp_return = np.mean(df['Sh_Return']) #np.mean 求Sh_Return的平均值
    print('Sh_Expected_Return = ')
    print(exp_return)
    df1.at[i,'Sh_Expected_Return'] = exp_return #寫入stock.txt
    print('\n')

    #求出每隻股票的效用函數 寫入stock.txt
    df1.at[i,'Utility'] = Ln_Weight*df1.loc[i,'Ln_Expected_Return']+(1-Ln_Weight)*df1.loc[i,'Ln_Expected_Return'] - 0.5*k*df1.loc[i,'Variance']#寫入stock.txt

    #計算Current_Asset_Ratio / EPS
    df1.at[i,'Current_Asset_Ratio'] = df1.loc[i,'Current_Assets'] / df1.loc[i,'Assets']
    df1.at[i,'EPS'] = 0.5* (df1.loc[i,'EPS_2016'] + df1.loc[i,'EPS_2017'])


sorted_df1 = df1.sort_values(by=['Utility','Current_Asset_Ratio','EPS'],ascending=False)
print(sorted_df1)
writer = pd.ExcelWriter('output.xlsx')
sorted_df1.to_excel(writer,'Sheet1')
#print('\n')
#print('1101_EPS_2017 = ')
#print(df1.loc[df1['Name'] == 1101])

