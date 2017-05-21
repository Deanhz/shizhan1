import pandas as pd
import numpy as np

datafile='./chapter7/demo/data/air_data.csv'
resultfile='./tmp/explore.xls'
data=pd.read_csv(datafile)

explore=data.describe().T #各个属性的统计
explore['null']=len(data)-explore['count'] #各个属性的空值数

explore=explore[['null','max','min']] #值关注空值数、最大值、最小值
explore.columns=['空值数','最大值','最小值']

explore.to_excel(resultfile) #保存

