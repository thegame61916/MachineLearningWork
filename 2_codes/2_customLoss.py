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
from functools import partial
from itertools import product

import time
import theano
import theano.tensor as T

batch_size = 128
nb_classes = 10
nb_epoch = 12

epsilon = 1.0e-9
def custom_objective(y_true, y_pred):
    '''Just another crossentropy'''
    y_pred = T.clip(y_pred, epsilon, 1.0 - epsilon)
    y_pred /= y_pred.sum(axis=-1, keepdims=True)
    cce = T.nnet.categorical_crossentropy(y_pred, y_true)
    return cce

weights = np.ones((10,10))
weights[1, 7] = 1.3
weights[7, 1] = 1.3
weights[5, 6] = 1.3
weights[6, 5] = 1.3
weights[2, 5] = 1.3
weights[5, 2] = 1.3
weights[3, 8] = 1.3
weights[8, 3] = 1.3
weights[0, 8] = 1.3
weights[8, 0] = 1.3
weights[6, 8] = 1.3
weights[8, 6] = 1.3

def w_categorical_crossentropy(y_true, y_pred):
    global weights
    nb_cl = len(weights)
    final_mask = K.zeros_like(y_pred[:, 0])
    y_pred_max = K.max(y_pred, axis=1)
    y_pred_max = K.reshape(y_pred_max, (K.shape(y_pred)[0], 1))
    y_pred_max_mat = K.equal(y_pred, y_pred_max)
    for c_p, c_t in product(range(nb_cl), range(nb_cl)):
        final_mask += (weights[c_t, c_p] * y_pred_max_mat[:, c_p] * y_true[:, c_t])
    return K.categorical_crossentropy(y_pred, y_true) * final_mask



inits1 = [w_categorical_crossentropy, custom_objective, 'categorical_crossentropy', 'mean_squared_error', 'mean_absolute_error', 'hinge', 'squared_hinge', 'cosine_proximity', 'poisson']
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
		                input_shape=input_shape))
	model.add(Activation('relu'))
	model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=pool_size))
	#model.add(Dropout(0.25))

	model.add(Flatten())
	model.add(Dense(128))
	model.add(Activation('relu'))
	#model.add(Dropout(0.5))
	model.add(Dense(nb_classes))
	model.add(Activation('softmax'))

	model.compile(loss=inits,
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
