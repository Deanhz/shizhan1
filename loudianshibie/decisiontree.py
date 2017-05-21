import pandas as pd
import numpy as np

datafile='./chapter6/demo/data/model.xls'
data=pd.read_excel(datafile)
data=data.as_matrix()

np.random.shuffle(data)
p=0.8
train=data[:int(len(data)*p),:]
test=data[int(len(data)*p):,:]

from sklearn.tree import DecisionTreeClassifier
treefile='./tmp/tree.pkl'
tree=DecisionTreeClassifier()
tree.fit(train[:,:3],train[:,3])

#保存模型
from sklearn.externals import joblib
joblib.dump(tree,treefile) 
#加载模型的方法:joblib.load(文件名)

from cm_plot import *
cm_plot(train[:,3],tree.predict(train[:,:3])).show()
