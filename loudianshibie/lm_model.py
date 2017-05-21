import numpy as np
import pandas as pd

datafile='./chapter6/demo/data/model.xls'#构造后特征的数据集
data=pd.read_excel(datafile)
data=data.as_matrix()
np.random.shuffle(data)#打乱数据
p=0.8
train=data[:int(len(data)*p),:]#80%训练集
test=data[int(len(data)*p):,:]#20%测试集

#构建LM神经网络模型
from keras.models import Sequential
from keras.layers.core import Dense,Activation

netfile='./tmp/net.model'#模型存储路径

net=Sequential()#建立神经网络
net.add(Dense(input_dim=3,output_dim=10))#添加输入层(3个节点)到隐藏层(10个节点)的连接
net.add(Activation('relu'))#隐藏层使用relu函数
net.add(Dense(input_dim=10,output_dim=1))#添加隐藏层(10节点)到输出层(1节点)的连接
net.add(Activation('sigmoid'))#输出层使用sigmoid函数
net.compile(loss='binary_crossentropy',optimizer='adam',class_mode='binary')#编译模型,使用adam方法求解

net.fit(train[:,:3],train[:,3],nb_epoch=100,batch_size=1)#训练模型
net.save_weights(netfile)#保存模型
predict_result=net.predict_classes(train[:,:3]).reshape(len(train))#预测结果变形

from cm_plot import *#导入自行编写的混淆矩阵可视化函数
cm_plot(train[:,3],predict_result).show()#显示混淆矩阵可视化结果
