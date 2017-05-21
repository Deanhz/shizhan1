import pandas as pd
import numpy as np
from scipy.interpolate import lagrange

inputfile='./chapter6/demo/data/missing_data.xls'
outputfile='./tmp/missing_data_processed.xls'

data=pd.read_excel(inputfile,header=None)
def ployinterp_columns(s,n,k=5):
    y=s[list(range(n-k,n))+list(range(n+1,n+1+k))]
    y=y[y.notnull()]
    return lagrange(y.index,list(y))(n)

for i in data.columns:
    for j in range(len(data)):
        if (data[i].isnull())[j]:
            data[i][j]=ployinterp_columns(data[i],j)
data.to_excel(outputfile,header=None,index=False)
