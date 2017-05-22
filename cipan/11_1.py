import pandas as pd
import numpy as np

discfile='./chapter11/demo/data/discdata.xls'
transformeddata='./tmp/discdata_processed.xls'
data=pd.read_excel(discfile)
data=data[data['TARGET_ID']==184].copy()

data_group=data.groupby('COLLECTTIME')
for i in data_group:
    print(i)
    
def attr_trans(x):
    result=pd.Series(index=['SYS_NAME','CWXT_DB:184:C:\\','CWXT_DB:184:D:\\','COLLECTTIME'])
    result['SYS_NAME']=x['SYS_NAME'].iloc[0]
    result['COLLECTTIME']=x['COLLECTTIME'].iloc[0]
    result['CWXT_DB:184:C:\\']=x['VALUE'].iloc[0]
    result['CWXT_DB:184:D:\\']=x['VALUE'].iloc[1]
    return result
data_processed=data_group.apply(attr_trans)
data_processed.to_excel(transformeddata,index=False)
