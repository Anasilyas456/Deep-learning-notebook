# import tensorflow as tf
# import numpy as np
# import seaborn as sns
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import StandardScaler,LabelEncoder
# from sklearn.impute import SimpleImputer
# from sklearn.model_selection import train_test_split
# import warnings
# warnings.filterwarnings('ignore')
# df=sns.load_dataset('titanic')
# p=df['alive']
# ##print(p.to_string())
# df=df.drop('deck',axis=1)
# df=df.drop('survived',axis=1)
# ### print(df.isnull().sum().sort_values(ascending=False))

# imputer=SimpleImputer(strategy='median')
# df['age']=imputer.fit_transform(df[['age']])
# imputerr=SimpleImputer(strategy='most_frequent')
# df['embarked']=imputerr.fit_transform(df[['embarked']]).ravel()
# df['embark_town']=imputerr.fit_transform(df[['embark_town']]).ravel()

# encoder = LabelEncoder()
# df['sex'] = encoder.fit_transform(df['sex'])
# df['embarked'] = encoder.fit_transform(df['embarked'])
# df['embark_town'] = encoder.fit_transform(df['embark_town'])
# df['class'] = encoder.fit_transform(df['class'])
# df['who'] = encoder.fit_transform(df['who'])
# df['alive'] = encoder.fit_transform(df['alive'])
# df['alone'] = encoder.fit_transform(df['alone'])

# x=df.drop(columns=['alive'])
# y=df['alive']
# scaler = StandardScaler()
# x = pd.DataFrame(scaler.fit_transform(x), columns=x.columns)

# xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=42)
# il=tf.keras.layers.Dense(10,activation='relu',input_shape=(xtrain.shape[1],))
# ol=tf.keras.layers.Dense(1,activation='sigmoid')
# model=tf.keras.models.Sequential([il,ol])
# model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

# model.fit(xtrain,ytrain,epochs=100,batch_size=32,verbose=0)  #####agar epoch show krne ho to verbose=1 ae ga or 0 tb ae ga jb show nhi krne
# loss,accuracy=model.evaluate(xtest,ytest,verbose=1)
# pred=model.predict(xtest)
# pred_binary = (pred > 0.5).astype(int)
# pred_labels = ['Yes' if p == 1 else 'No' for p in pred_binary]
# print(pred_labels)
# print(accuracy)


#import libraries
import tensorflow as tf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.impute import SimpleImputer

#import warnings
import warnings
warnings.filterwarnings('ignore')
#load data
data=sns.load_dataset('titanic')
# ## print(data.head())
# ## print(data.isnull().sum().sort_values(ascending=False))

#cleaning the data
df=data.drop(['age','deck','alive'],axis=1)

df['embarked']=df['embarked'].fillna(df['embarked'].mode()[0])
df['embark_town']=df['embark_town'].fillna(df['embark_town'].mode()[0])
#print(df.isnull().sum().sort_values(ascending=False))

#encoding the data
encoder=LabelEncoder()
for col in df.columns:
    if df[col].dtypes=='object' or df[col].dtypes.name=='category':
        df[col]=encoder.fit_transform(df[col])

#train the data
x=df.drop('survived',axis=1)
y=df['survived']
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=42)

#scale the data
scaler=StandardScaler()
xtscaled=scaler.fit_transform(xtrain)
xscaled=scaler.transform(xtest) 

#make a neural network
il=tf.keras.layers.Dense(10,activation='relu',input_shape=(xtscaled.shape[1],))
ol=tf.keras.layers.Dense(1,activation='sigmoid')
mod=tf.keras.models.Sequential([il,ol])
mod.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
mod.fit(xtscaled,ytrain,epochs=100,batch_size=32,verbose=1)
accuracy=mod.evaluate(xscaled,ytest,verbose=1)
print(accuracy)
