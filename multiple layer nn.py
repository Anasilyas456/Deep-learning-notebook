# #import libraries
# import pandas as pd 
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import StandardScaler,LabelEncoder
# import tensorflow as tf
# import seaborn as sns
# from sklearn.model_selection import train_test_split 

# #load data
# data=sns.load_dataset('titanic')
# print(data.head())
# print(data.isnull().sum().sort_values(ascending=False))

# #cleaning the data
# df=data.drop(['age','deck','alive'],axis=1)
# df['embarked']=df['embarked'].fillna(df['embarked'].mode()[0])
# df['embark_town']=df['embark_town'].fillna(df['embark_town'].mode()[0])
# print(df.isnull().sum().sort_values(ascending=False))
# #encoding the data
# encoder=LabelEncoder()
# for col in df.columns:
#     if df[col].dtypes=='object' or df[col].dtypes.name=='category':
#         df[col]=encoder.fit_transform(df[col])
# #train the data
# x=df.drop('survived',axis=1)
# y=df['survived']
# xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=42)
# print(xtrain.shape) #to find the min layers that we can put

# #scaling the data
# scaler=StandardScaler()
# xtscale=scaler.fit_transform(xtrain)
# xtestscale=scaler.transform(xtest)
# #build a multiple NN model
# il=tf.keras.layers.Dense(13,activation='relu',input_shape=(xtscale.shape[1],))
# hl=tf.keras.layers.Dense(7,activation='relu')
# ol=tf.keras.layers.Dense(1,activation='sigmoid')
# model=tf.keras.models.Sequential([il,hl,ol])
# #compile a model
# model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
# model.fit(xtscale,ytrain,epochs=100,batch_size=32,verbose=1)
# #evaluate a model
# accuracy=model.evaluate(xtestscale,ytest,verbose=1)
# print(accuracy)


#*************call back function for early epochs********************
#import libraries
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler,LabelEncoder
import tensorflow as tf
import seaborn as sns
from sklearn.model_selection import train_test_split 

#load data
data=sns.load_dataset('titanic')
print(data.head())
print(data.isnull().sum().sort_values(ascending=False))

#cleaning the data
df=data.drop(['age','deck','alive'],axis=1)
df['embarked']=df['embarked'].fillna(df['embarked'].mode()[0])
df['embark_town']=df['embark_town'].fillna(df['embark_town'].mode()[0])
print(df.isnull().sum().sort_values(ascending=False))
#encoding the data
encoder=LabelEncoder()
for col in df.columns:
    if df[col].dtypes=='object' or df[col].dtypes.name=='category':
        df[col]=encoder.fit_transform(df[col])
#train the data
x=df.drop('survived',axis=1)
y=df['survived']
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=42)
print(xtrain.shape) #to find the min layers that we can put

#scaling the data
scaler=StandardScaler()
xtscale=scaler.fit_transform(xtrain)
xtestscale=scaler.transform(xtest)
#build a multiple NN model
il=tf.keras.layers.Dense(13,activation='relu',input_shape=(xtscale.shape[1],))
hl=tf.keras.layers.Dense(7,activation='relu')
ol=tf.keras.layers.Dense(1,activation='sigmoid')
model=tf.keras.models.Sequential([il,hl,ol])
#make a call back functions
from tensorflow.keras.callbacks import EarlyStopping
early_stop=EarlyStopping(patience=5)
#compile a model
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(xtscale,ytrain,epochs=100,batch_size=32,verbose=1,callbacks=[early_stop])
#evaluate a model
accuracy=model.evaluate(xtestscale,ytest,verbose=1)
print(accuracy)
