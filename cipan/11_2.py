import pandas as pd
import numpy as np

discfile='./chapter11/demo/data/discdata_processed.xls'
data=pd.read_excel(discfile,index_col='COLLECTTIME')
train_data=data.iloc[:len(data)-5]['CWXT_DB:184:C:\\']

#平稳性检验
from statsmodels.tsa.stattools import adfuller
diff=0
adf=adfuller(train_data) #ADF检验，adf[1]保存的是p值
while adf[1]>=0.05:
    diff=diff+1
    adf=adfuller(train_data.diff(diff).dropna())
print('原始序列经过%s次差分后平稳，p值为%s'%(diff,adf[1]))
#原始序列经过1次差分后平稳，p值为9.57297559233e-07

#白噪声检验
from statsmodels.stats.diagnostic import acorr_ljungbox
([lb],[p])=acorr_ljungbox(train_data,lags=1)
if p<0.05:
    print('原始序列为非白噪声序列')
else:
    print('原始序列为白噪声序列')
([lb],[p])=acorr_ljungbox(train_data.diff(2).dropna(),lags=1)
adf=adfuller(train_data.diff(2).dropna())
#以上结果发现，经过两阶差分后，序列为平稳非白噪声序列

#定阶
from statsmodels.tsa.arima_model import ARIMA
pmax=int(len(train_data)/10)
qmax=int(len(train_data)/10)
bic_matrix=[]

for p in range(pmax):
    tmp=[]
    for q in range(qmax):
        try:
            tmp.append(ARIMA(train_data,(p,2,q)).fit().bic)
        except:
            tmp.append(None)
    bic_matrix.append(tmp)
bic_matrix=pd.DataFrame(bic_matrix)
p,q=bic_matrix.stack().idxmin()
print('BIC最小p值和q值为:%s,%s'%(p,q))
#BIC最小p值和q值为:1,1

#建模
arima=ARIMA(train_data,(1,2,1)).fit()

#模型确定后，看预测的残差序列是否为白噪声，如果不是，说明残差中还存在有用的信息，需要进一步提取
train_pred=arima.predict(typ='levels')
pred_error=(train_pred-train_data).dropna()
lb,p=acorr_ljungbox(pred_error,lags=12)
h=(p<0.05).sum()
if h>0:
    print('模型ARIMA(1,2,1)不符合白噪声检验')
else:
    print('模型ARIMA(1,2,1)符合白噪声检验')
#模型ARIMA(1,2,1)符合白噪声检验

#模型评价
discfile='./chapter11/demo/data/discdata_processed.xls'
data=pd.read_excel(discfile,index_col='COLLECTTIME')
test_data=data.iloc[len(data)-5:]['CWXT_DB:184:C:\\']

#计算训练误差
abs_=(train_pred-train_data).dropna().abs()
mae_=abs_.mean() #平均绝对误差
rmse_=((abs_**2).mean())**0.5 #均方根误差
mape_=(abs_/train_data).dropna().mean() #平均绝对百分误差
print('平均绝对误差:%0.4f'%mae_)
print('均方根误差:%0.4f'%rmse_)
print('平均绝对百分误差:%0.4f'%mape_)
#平均绝对误差:178309.1452
#均方根误差:349367.3956
#平均绝对百分误差:0.0052

test_pred=arima.predict(start='2014-11-12',end='2014-11-16',typ='levels')#预测结果
#计算误差
abs_=(test_pred-test_data).abs()
mae_=abs_.mean() #平均绝对误差
rmse_=((abs_**2).mean())**0.5 #均方根误差
mape_=(abs_/test_data).mean() #平均绝对百分误差
print('平均绝对误差:%0.4f'%mae_)
print('均方根误差:%0.4f'%rmse_)
print('平均绝对百分误差:%0.4f'%mape_)

##下面是用一阶差分建模测试（一阶差分序列是平稳白噪声序列，理论上不能进行建模和预测的）
pmax=int(len(train_data)/10)
qmax=int(len(train_data)/10)
bic_matrix=[]
for p in range(pmax):
    tmp=[]
    for q in range(qmax):
        try:
            tmp.append(ARIMA(train_data,(p,1,q)).fit().bic)
        except:
            tmp.append(None)
    bic_matrix.append(tmp)
bic_matrix=pd.DataFrame(bic_matrix)
p,q=bic_matrix.stack().idxmin()
print('BIC最小p值和q值为:%s,%s'%(p,q))
#BIC最小p值和q值为:0,0

#建模
arima=ARIMA(train_data,(0,1,0)).fit()
#建模后的预测差分序列是否是白噪声
train_pred=arima.predict(typ='levels')
pred_error=(train_pred-train_data).dropna()
lb,p=acorr_ljungbox(pred_error,lags=12)
h=(p<0.05).sum()
if h>0:
    print('模型ARIMA(0,1,0)不符合白噪声检验')
else:
    print('模型ARIMA(0,1,0)符合白噪声检验')
#模型ARIMA(0,1,0)符合白噪声检验

#计算训练误差
abs_=(train_pred-train_data).dropna().abs()
mae_=abs_.mean() #平均绝对误差
rmse_=((abs_**2).mean())**0.5 #均方根误差
mape_=(abs_/train_data).dropna().mean() #平均绝对百分误差
print('平均绝对误差:%0.4f'%mae_)
print('均方根误差:%0.4f'%rmse_)
print('平均绝对百分误差:%0.4f'%mape_)

test_pred=arima.predict(start='2014-11-12',end='2014-11-16',typ='levels')
#计算测试误差
abs_=(test_pred-test_data).abs()
mae_=abs_.mean() #平均绝对误差
rmse_=((abs_**2).mean())**0.5 #均方根误差
mape_=(abs_/test_data).mean() #平均绝对百分误差
print('平均绝对误差:%0.4f'%mae_)
print('均方根误差:%0.4f'%rmse_)
print('平均绝对百分误差:%0.4f'%mape_)
##通过实验发现，用一阶差分得到平稳白噪声序列，居然也能预测。。




