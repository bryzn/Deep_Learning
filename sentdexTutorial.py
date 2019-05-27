import tensorflow as tf
from keras.layers import Dense,Flatten
from keras.utils import normalize
from keras.models import Sequential, load_model
import numpy as np
import matplotlib.pyplot as plt
#printing the version 1.12.0
print(tf.__version__)
#mnist is handwritten number data
mnist = tf.keras.datasets.mnist

#x_train = features of the 28x28 pixels
#y_train = is the label (1,2,3,..)
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#normalizing data
x_train = normalize(x_train, axis=1)
x_test = normalize(x_test, axis=1)

#prints the image in black/white
print(x_train[0])
plt.imshow(x_train[0],cmap=plt.cm.binary)
#plt.show()

#showing the value
print(y_train[0])

#building our model in a sequential order (going forward)
model = Sequential()
#making image flat to fit into our input layer
model.add(Flatten())
#will be using simple dense layer
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))

#output layer
model.add(Dense(10, activation='softmax')
)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(x_train, y_train, epochs=3)

model.summary()
#testing from out of sample value
val_loss, val_acc = model.evaluate(x_test, y_test)
print(f'Testing Loss: {val_loss}')
print(f'Testing Accuracy: {val_acc}')
#saving model
model.save('epic_num_reader.model')
new_model = load_model('epic_num_reader.model', compile=False)

#making predictions
predictions = new_model.predict(x_test)
print(predictions) #in probablity distributions
print(np.argmax(predictions[0])) #prints 7

plt.imshow(x_test[0], cmap=plt.cm.binary)
plt.show()






























