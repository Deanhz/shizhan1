import pandas as pd
import numpy as np

inputfile='./chapter10/demo/data/water_heater.xls'
data=pd.read_excel(inputfile)
threshold=pd.Timedelta(minutes=5)

n=4
data['发生时间']=pd.to_datetime(data['发生时间'],format='%Y%m%d%H%M%S')
data=data[data['水流量']>0]

def event_num(ts):
    d=data['发生时间'].diff()>ts
    return d.sum()+1
dt=[pd.Timedelta(minutes=i) for i in np.arange(1,9,0.25)]
h=pd.DataFrame(dt,columns=['阈值'])
h['事件数']=h['阈值'].apply(event_num)

h['斜率']=h['事件数'].diff()/0.25
h['斜率指标']=pd.rolling_mean(h['斜率'].abs(),n)

ts=h['阈值'][h['斜率指标'].idxmin()-n]
if ts>threshold:
    ts=pd.Timedelta(minutes=4)

x=np.arange(2,9,0.25)
y=h.ix[4:,'事件数'].values
import matplotlib.pyplot as plt
plt.figure()
plt.plot(x,y)
plt.show()
