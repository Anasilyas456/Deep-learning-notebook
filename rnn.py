#import libraries
import numpy as np 
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import SimpleRNN
#generate data
x=np.random.rand(100,10,1)
y=np.random.randint(0,2,size=(100,))
#define the model
model=tf.keras.Sequential([
    SimpleRNN(32,input_shape=(10,1)),
    tf.keras.layers.Dense(1,activation='sigmoid')
    
])
#compile the model
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
mod=model.fit(x,y,epochs=10,batch_size=32,verbose=1)

print(mod)
