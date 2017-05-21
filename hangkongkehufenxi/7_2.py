import pandas as pd
import numpy as np

datafile='./chapter7/demo/data/air_data.csv'
cleanedfile='./tmp/data_cleaned.xls'
data=pd.read_csv(datafile)

data=data[data['SUM_YR_1'].notnull()&data['SUM_YR_2'].notnull()]#只保留票价非空的记录
#只保留票价非零，或平均折扣率与总飞行公里数同时为0的记录
index1=data['SUM_YR_1']!=0
index2=data['SUM_YR_2']!=0
index3=(data['SEG_KM_SUM']==0)&(data['avg_discount']==0)
data=data[index1|index2|index3]

data.to_excel(cleanedfile)

