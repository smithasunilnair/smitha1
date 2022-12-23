import tensorflow
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense,Flatten

(X_train,y_train),(X_test,y_test) = keras.datasets.mnist.load_data() # 70,000 images  of built-in data MNIST

no_of_images_train = len(X_train)
no_of_images_test = len(X_test)

print('no_of_images_train = ',no_of_images_train)
print('no_of_images_test = ',no_of_images_test)

X_train.shape # 3D array data  # 60,000 images each of size 28 x 28

X_train[0]   # to display the 1st image

X_train[0].shape  # 1st image is of size 28 x 28   similary all images

X_test.shape  # 3D array data # 10,000 images each of size 28 x 28

y_train   # contains labels  like the 1st image is a 5, 2nd 0, 3rd 4 etc

import matplotlib.pyplot as plt
plt.imshow(X_train[0])   #change the index and try a few more to visualize the digits

#pixel values range from 0 to 255 in a grayscale image --scaling
#So to bring the pixel values to a particular range (0,1), perform the following
X_train = X_train/255
X_test = X_test/255

X_train[0]  # try another row also to see the range of values

model = Sequential()

model.add(Flatten(input_shape=(28,28)))   # to convert the 28 rows into 1 single row 
 # flatten layer is used to convert high dim to a single dim, so no trainable parameters
model.add(Dense(128,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(10,activation='softmax'))   # multiclass classification; so softmax is used

model.summary()
# 28 * 28 = 784 inputs going to 128 dense layer  - so 784 * 128 = 100352
# plus 128 biases  #100352 + 128 = 100480

model.compile(loss='sparse_categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])
#https://www.tensorflow.org/api_docs/python/tf/keras/losses  for more information

MyModel = model.fit(X_train,y_train,epochs=10,validation_split=0.2)

model.predict(X_test)   # result is of 10,000 test images

"""Compare y_pred and y_test values one by one
"""

print (y_pred[0])     # the value 7 has the max probability value from the output

print(y_test[0])     # test result of 1st image is equal to that of y_pred

#check the overall accuracy between y_pred and y_test

from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)

#compare the training and validation loss
plt.plot(MyModel.history['loss'], color ='blue')
plt.plot(MyModel.history['val_loss'], color = 'red')

#compare the training and validation accuracy
plt.plot(MyModel.history['accuracy'],color ='blue')
plt.plot(MyModel.history['val_accuracy'],color ='red')

# Even though training acc is very good, validation acc is not that good compared to that of training.
# Hence there is a condition of overfitting (however not severe) -- we can think of regularization techniques to overcome this.

#printing confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print (cm)

# focus on diagnols -- 971 test image digits are correctly predicted as 0, 1129 test data was correctly predicted as 1 etc  -- value differs
# however there are a few errors

from sklearn import metrics
print('Precision = ',metrics.precision_score(y_test,y_pred, average =None))
print ('*********')
print('Recall = ', metrics.recall_score(y_test,y_pred,average =None))
print ('*********')
print('F1-score = ', metrics.f1_score(y_test,y_pred,average =None))

from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))