import pandas as pd
import numpy as np

inputfile1='./chapter10/demo/data/train_neural_network_data.xls'
inputfile2='./chapter10/demo/data/test_neural_network_data.xls'

testoutputfile='./tmp/test_output_data.xls'
data_train=pd.read_excel(inputfile1)
data_test=pd.read_excel(inputfile2)

x_train=data_train.iloc[:,5:17].as_matrix()
y_train=data_train.iloc[:,4].as_matrix()
x_test=data_test.iloc[:,5:17].as_matrix()
y_test=data_test.iloc[:,4].as_matrix()

from keras.models import Sequential
from keras.layers.core import Dense,Dropout,Activation

model=Sequential()
model.add(Dense(input_dim=11,output_dim=17))
model.add(Activation('relu'))

model.add(Dense(input_dim=17,output_dim=10))
model.add(Activation('relu'))

model.add(Dense(input_dim=10,output_dim=1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam',class_mode='binary')

model.fit(x_train,y_train,nb_epoch=100,batch_size=1)

#model.save_weights('./tmp/net.model')

r=pd.DataFrame(model.predict_classes(x_test),columns=['预测结果'])
pd.concat([data_test.iloc[:,:5],r],axis=1).to_excel(testoutputfile)

