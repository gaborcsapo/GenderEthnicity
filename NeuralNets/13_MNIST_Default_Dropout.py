
# coding: utf-8

# In[1]:


from __future__ import print_function

import double_log
def print(*args, **kwargs):
    return double_log.print(*args, **kwargs)

import pickle
import pandas as pd
import numpy as np

import keras
from keras.engine.topology import Layer
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Softmax, Reshape
from keras.layers import Conv2D, Conv1D, MaxPooling2D
from keras import backend as K
import plot_conf_matrix as pcm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import operator
import tensorflow as tf

print("=====IMPORTING|||SCRIPT STARTS|||LOGGING PURPOSE======")


# In[2]:


with open('../../Images/MNIST_noisy/noisy_mnist_20.pickle', 'rb') as fp:
    mnist_noise = pickle.load(fp)
print(mnist_noise.keys())


# In[3]:


batch_size = 10
num_classes = 10
epochs = 1

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
# the data, split between train and test sets
y_train = mnist_noise['train']
y_test = mnist_noise['test']

(x_train, y_train_clean), (x_test, y_test_clean) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

print(np.sum(y_train == y_train_clean), " elems are true in train")
print(np.sum(y_test == y_test_clean), " elems are true in test")
#pcm.plot_confusion_matrix(y_train_clean, y_train, normalize=True)

cm = confusion_matrix(y_train_clean, y_train)
cm = confusion_matrix(y_train_clean, y_train).astype('float') / cm.sum(axis=1)[:, np.newaxis]

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
y_test_clean = keras.utils.to_categorical(y_test_clean, num_classes)


# In[4]:


np.sum(cm, axis=0).shape


# In[25]:


## SWITCH BETWEEN NOISY AND CLEAN DATA
#y_test = y_test_clean

## CUSTOM WEIGHT LAYER FROM CONF MATRIX
def confusion_kernel(shape):
    return cm   

class MyLayer(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel', 
                                      shape=input_shape,
                                      initializer=confusion_kernel,
                                      trainable=True)
        print('inputsh', input_shape)
        super(MyLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        print(x.shape)
        print('softmax',tf.nn.softmax(tf.einsum('bn,nm->bn',x, self.kernel)).shape)
        #return tf.nn.softmax(tf.einsum('n,nm->n',x[0][0], self.kernel))
        return tf.einsum('bn,nm->bn',x, self.kernel)

    def compute_output_shape(self, input_shape):
        print('outputshpae: ', (input_shape[0], self.output_dim))
        return (1, self.output_dim)


# In[27]:


K.set_learning_phase(1)
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

#model.add(Reshape((1, num_classes,)))
model.add(MyLayer(num_classes))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=[keras.metrics.categorical_accuracy])

print("====Train start======")
print_callback = keras.callbacks.LambdaCallback(
    on_epoch_end=lambda epoch,logs: double_log.logger.debug('epoch: '+ str(epoch+1)+ ' logs: '+ str(logs)))

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test),
          callbacks=[print_callback])
pred = model.predict(x_test, verbose=1, batch_size=10)

print('F1-score:', f1_score([max(enumerate(i), key=operator.itemgetter(1))[0] for i in pred.tolist()], [max(enumerate(i), key=operator.itemgetter(1))[0] for i in y_test_clean.tolist()], average='micro'))
print("====Train end======")


# In[19]:


inp = model.input                                           # input placeholder
outputs = [layer.output for layer in model.layers]          # all layer outputs
functor = K.function([inp]+ [K.learning_phase()], outputs ) # evaluation function

# Testing
testp = x_test[0:1]
layer_outs = functor([testp, 1.])
print (layer_outs)


# In[22]:


pred = model.predict(x_test[0:10], verbose=1, batch_size=10)
print(pred)


# In[16]:


x_test.shape


# In[ ]:


model.layers.pop()
model2 = Model(model.input, model.layers[-1].output)


# In[ ]:


pred2 = model2.predict(x_test, verbose=1, batch_size=10)
print(pred2.shape)
print('F1-score:', f1_score([max(enumerate(i), key=operator.itemgetter(1))[0] for i in pred2.tolist()], [max(enumerate(i), key=operator.itemgetter(1))[0] for i in y_test_clean.tolist()], average='micro'))


# In[ ]:


model.summary()


# ### clean data: 0.9874
# ### 20% noisy data: 0.9858
# ### 30% noisy data: 0.9843
# ### 40% noisy data: 0.9808
# ### 50% noisy data: 0.9785
# ### 70% noisy data: 0.9492
# ### 80% noisy data: 0.8506
