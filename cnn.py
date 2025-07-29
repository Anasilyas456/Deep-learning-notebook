#import libraries
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ##load the data
df=tf.keras.datasets.mnist.load_data()
(xtrain,ytrain),(xtest,ytest)=df
###show some images of data
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(xtrain[i],cmap=plt.cm.binary)
    plt.ylabel(ytrain[i])
plt.show()
##explore the data
print()
print(f"we have{len(xtrain)} images in training data and {len(xtest)} images in testing data")
print(xtrain.shape,xtest.shape)
##display value of each pixel
for row in xtrain[0]:
    for pixel in row:
        print("{:3}".format(pixel),end=" ")
    print()

#####now we con reshape the data as we have 3 value(color)
xtrain=xtrain.reshape(xtrain.shape +(1,))
ytrain=ytrain.reshape(ytrain.shape +(1,))

###normalize the data to standard value
xtrain=xtrain/255
ytrain=ytrain/255
###now change the datatype to float
xtrain=xtrain.astype('float32')
ytrain=ytrain.astype('float32')
######now check the pixel in standard value
for row in xtrain[0, : , : ,0]:
    for pixel in row:
        print("{:.2f}".format(pixel),end=" ")
    print()

## now build the NN model

model=tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=32,kernel_size=(3,3),activation='relu',input_shape=(28,28,1)),
    tf.keras.layers.Conv2D(filters=32,kernel_size=(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(filters=32,kernel_size=(3,3),activation='relu'),
    tf.keras.layers.Conv2D(filters=32,kernel_size=(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dense(10,activation='softmax')
])
#####now we compile the model

model.compile(optimizer='adam',loss='SparseCategoricalCrossentropy',metrics=['accuracy'])
######train the model
model.fit(xtrain,ytrain,epochs=20,batch_size=32,verbose=1)
####evaluate the model
accuracy,loss=model.evaluate(xtest,ytest)
print(accuracy)
