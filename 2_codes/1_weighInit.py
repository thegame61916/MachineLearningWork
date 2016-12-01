'''Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

#from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility
import os
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
import time

batch_size = 128
nb_classes = 10
nb_epoch = 12
def random01(shape, name=None):
    value = np.random.random(shape)
    #print value
    return K.variable(value, name=name)

def random11(shape, name=None):
    value = 2*np.random.random(shape) - 1
    #print value
    return K.variable(value, name=name)

def randomBig(shape, name=None):
    value = 2000*np.random.random(shape) - 1000
    #print value
    return K.variable(value, name=name)

def constant(shape, name=None):
    value = np.random.random(shape)
    value = value - value + 1.5; 
    #print value
    return K.variable(value, name=name)

def random01Small(shape, name=None):
    value = np.random.random(shape) * 0.05
    #print value
    return K.variable(value, name=name)

def random11Small(shape, name=None):
    value = (2*np.random.random(shape) - 1) * 0.05
    #print value
    return K.variable(value, name=name)

def constantSmall(shape, name=None):
    value = np.random.random(shape)
    value = value - value + 0.05; 
    #print value
    return K.variable(value, name=name)

inits1 = [ random01Small, random11Small, constantSmall, random01, random11, randomBig, constant, 'zero', 'uniform', 'normal', 'glorot_normal', 'glorot_uniform', 'he_normal']
tm = []
loss= []
accu = []

for inits in inits1:
	start_time = time.time()	
	print 
	print
	print 'value is: ', inits
	print
	# input image dimensions
	img_rows, img_cols = 28, 28
	# number of convolutional filters to use
	nb_filters = 32
	# size of pooling area for max pooling
	pool_size = (2, 2)
	# convolution kernel size
	kernel_size = (3, 3)

	# the data, shuffled and split between train and test sets
	(X_train, y_train), (X_test, y_test) = mnist.load_data()

	if K.image_dim_ordering() == 'th':
	    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
	    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
	    input_shape = (1, img_rows, img_cols)
	else:
	    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
	    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
	    input_shape = (img_rows, img_cols, 1)

	X_train = X_train.astype('float32')
	X_test = X_test.astype('float32')
	X_train /= 255
	X_test /= 255
	#print('X_train shape:', X_train.shape)
	#print(X_train.shape[0], 'train samples')
	#print(X_test.shape[0], 'test samples')

	# convert class vectors to binary class matrices
	Y_train = np_utils.to_categorical(y_train, nb_classes)
	Y_test = np_utils.to_categorical(y_test, nb_classes)

	model = Sequential()

	model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
		                border_mode='valid',
		                input_shape=input_shape, init = inits))
	model.add(Activation('relu'))
	model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1], init = inits))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=pool_size))
	#model.add(Dropout(0.25))

	model.add(Flatten())
	model.add(Dense(128, init = inits))
	model.add(Activation('relu'))
	#model.add(Dropout(0.5))
	model.add(Dense(nb_classes, init = inits))
	model.add(Activation('softmax'))

	model.compile(loss='categorical_crossentropy',
		      optimizer='adadelta',
		      metrics=['accuracy'])

	history = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
		  verbose=0, validation_data=(X_test, Y_test))
	score = model.evaluate(X_test, Y_test, verbose=0)
	print 'acc=', history.history['acc']
	print 'val_acc=', history.history['val_acc']
	print 'loss=', history.history['loss']
	print 'val_loss=', history.history['val_loss']
	loss.append(score[0])
        accu.append(score[1])
	tm.append(time.time() - start_time)
print
print 'test_score=', loss
print 
print 'test_accuracy=', accu
print
print "time_taken:", tm
