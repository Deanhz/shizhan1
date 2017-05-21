import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
inputfile='./tmp/zscoreddata.xls'
k=5
data=pd.read_excel(inputfile)

#聚类
kmodel=KMeans(n_clusters=k,n_jobs=4)
kmodel.fit(data)

kmodel.cluster_centers_ #聚类的中心
kmodel.labels_ #各样本对应的类别

labels=pd.Series(kmodel.labels_)
labels.value_counts()#各类别统计

data_labels=pd.concat([data,labels],axis=1)
data_labels.columns=list(data.columns)+['类别']

clusters=pd.DataFrame(kmodel.cluster_centers_)#聚类中心
clusters.columns=data.columns

outputfile='./tmp/data_clusters.xls'
clusters.to_excel(outputfile)
