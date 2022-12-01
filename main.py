import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import keras
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
X_train.shape,y_train.shape
X_test.shape, y_test.shape
X_train[0]
y_train[0]
plt.imshow(X_train[0],cmap='Greys')
plt.imshow(X_train[1],cmap='Greys')
class_labels = ["T-shirt/top","Trouser",	"Pullover","Dress","Coat",	"Sandal",	"Shirt",	"Sneaker",	"Bag",	"Ankle boot"]
plt.figure(figsize=(16,16))
j=1
for i in np.random.randint(0,1000,25):
    plt.subplot(5,5,j); j+=1
    plt.imshow(X_train[i],cmap="Greys")
    plt.axis('off')
    plt.title('{} / {}'.format(class_labels[y_train[i]],y_train[i]))
X_train.shape
X_train.ndim  #no. of dimensions
X_train = np.expand_dims(X_train,-1) 
X_train.shape
X_train.ndim
X_train[0]
X_train = X_train/255
X_test= X_test/255
from sklearn.model_selection import train_test_split
X_train,X_validation,y_train,y_validation=train_test_split(X_train,y_train,test_size=0.2,random_state=2020)
X_train.shape,X_validation.shape,y_train.shape,y_validation.shape
model=keras.models.Sequential([
                         keras.layers.Conv2D(filters=32,kernel_size=3,strides=(1,1),padding='valid',activation='relu',input_shape=[28,28,1]),
                         keras.layers.MaxPooling2D(pool_size=(2,2)),
                         keras.layers.Flatten(),
                         keras.layers.Dense(units=128,activation='relu'),
                         keras.layers.Dense(units=10,activation='softmax')
])
model.summary()
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(X_train,y_train,epochs=10,batch_size=512,verbose=1,validation_data=(X_validation,y_validation))
X_test = np.expand_dims(X_test,-1)
y_pred = model.predict(X_test)
y_pred.round(2)
y_test
model.evaluate(X_test, y_test)
plt.figure(figsize=(16,16))

j=1
for i in np.random.randint(0,1000,25):
    plt.subplot(5,5,j); j+=1
    plt.imshow(X_test[i].reshape(28,28),cmap='Greys')
    plt.title('Actual= {} / {} \npredicted = {} /{}'.format(class_labels[y_test[i]],y_test[i],class_labels[np.argmax(y_pred[i])],np.argmax(y_pred[i]))) 
    plt.axis('off')
from sklearn.metrics import confusion_matrix
plt.figure(figsize=(16,9))
y_pred_labels = [ np.argmax(label) for label in y_pred]
cm= confusion_matrix(y_test,y_pred_labels)
sns.heatmap(cm, annot=True , fmt='d', xticklabels=class_labels, yticklabels=class_labels)

from sklearn.metrics import classification_report
cr= classification_report(y_test,y_pred_labels,target_names=class_labels)
print(cr) 
from google.colab import drive
drive.mount('/content/drive')
model.save('fashion_mnsit_cnn_model.h5')
    
