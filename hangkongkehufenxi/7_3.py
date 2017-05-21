import pandas as pd
import numpy as np

datafile='./chapter7/demo/data/zscoredata.xls'
zscoredfile='./tmp/zscoreddata.xls'
data=pd.read_excel(datafile)

#标准化处理
data=(data-data.mean())/data.std()
data.columns=['Z'+i for i in data.columns]

data.to_excel(zscoredfile,index=False)
