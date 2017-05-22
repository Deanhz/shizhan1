import pandas as pd
import numpy as np

inputfile='./chapter10/demo/data/water_heater.xls'
outputfile='./tmp/dividsequence.xls'
data=pd.read_excel(inputfile)

threshold=pd.Timedelta(minutes=4)
data['发生时间']=pd.to_datetime(data['发生时间'],format='%Y%m%d%H%M%S')
data=data[data['水流量']>0]
d=data['发生时间'].diff()>threshold
data['事件编号']=d.cumsum()+1

data.to_excel(outputfile)

