import numpy as np
import os
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=cpu,floatX=float32"
import theano
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import cifar10
from keras.models  import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
import matplotlib.pyplot as plt

##########################################################################################################################################
#Shifting
##########################################################################################################################################
print '##################################################################################################################################'
print 'shifting'
print '##################################################################################################################################'
cifar10 = np.load('data/lab2/lab2_data/cifar10_data.npz')
X_train = cifar10['X_train']
y_train = cifar10['y_train']
X_test = cifar10['X_test']
y_test = cifar10['y_test']

#print "Training data:"
#print "Number of examples: ", X_train.shape[0]
#print "Number of channels:",X_train.shape[1] 
#print "Image size:", X_train.shape[2], X_train.shape[3]
#print
#print "Test data:"
#print "Number of examples:", X_test.shape[0]
#print "Number of channels:", X_test.shape[1]
#print "Image size:",X_test.shape[2], X_test.shape[3] 

plot = []
for i in range(1,10):
    plot_image = X_train[100*i,:,:,:].transpose(1,2,0)
    for j in range(1,10):
	plot_image = np.concatenate((plot_image, X_train[100*i+j,:,:,:].transpose(1,2,0)), axis=1)
    if i==1:
	plot = plot_image
    else:
	plot = np.append(plot, plot_image, axis=0)

#plt.imshow(plot)
#plt.axis('off')
#plt.show()

#print "mean before normalization:", np.mean(X_train) 
#print "std before normalization:", np.std(X_train)

mean=[0,0,0]
std=[0,0,0]
newX_train = np.ones(X_train.shape)
newX_test = np.ones(X_test.shape)
for i in xrange(3):
    mean[i] = np.mean(X_train[:,i,:,:])
    std[i] = np.std(X_train[:,i,:,:])
    
for i in xrange(3):
    newX_train[:,i,:,:] = X_train[:,i,:,:] - mean[i]
    newX_train[:,i,:,:] = newX_train[:,i,:,:] / std[i]
    newX_test[:,i,:,:] = X_test[:,i,:,:] - mean[i]
    newX_test[:,i,:,:] = newX_test[:,i,:,:] / std[i]
	
    
X_train = newX_train
X_test = newX_test

#print "mean after normalization:", np.mean(X_train)
#print "std after normalization:", np.std(X_train)

batchSize = 50                    #-- Training Batch Size
num_classes = 10                  #-- Number of classes in CIFAR-10 dataset
num_epochs = 10                   #-- Number of epochs for training   
learningRate= 0.001               #-- Learning rate for the network
lr_weight_decay = 0.95            #-- Learning weight decay. Reduce the learn rate by 0.95 after epoch


img_rows, img_cols = 32, 32       #-- input image dimensions

Y_train = np_utils.to_categorical(y_train, num_classes)
Y_test = np_utils.to_categorical(y_test, num_classes)


d = len(X_train[0])
c = len(X_train[0][0][0])
r = len(X_train[0][0])
X_flip = []
print 'Shifts Started'
datagen = ImageDataGenerator(width_shift_range = 0.2, height_shift_range = 0.2)
X_trans = X_train[:]
datagen.fit(X_trans)
np.append(X_train, X_trans)
Y_trans = Y_train[:]
np.append(Y_train, Y_trans)

print 'Shifts done'

from keras import backend as K
K.set_image_dim_ordering('th')

model = Sequential()                                                #-- Sequential container.

model.add(Convolution2D(6, 5, 5,                                    #-- 6 outputs (6 filters), 5x5 convolution kernel
	                border_mode='valid',
	                input_shape=(3, img_rows, img_cols)))       #-- 3 input depth (RGB)
model.add(Activation('relu'))                                       #-- ReLU non-linearity 
model.add(MaxPooling2D(pool_size=(2, 2)))                           #-- A max-pooling on 2x2 windows
model.add(Convolution2D(16, 5, 5))                                  #-- 16 outputs (16 filters), 5x5 convolution kernel
model.add(Activation('relu'))                                       #-- ReLU non-linearity
model.add(MaxPooling2D(pool_size=(2, 2)))                           #-- A max-pooling on 2x2 windows

model.add(Flatten())                                                #-- eshapes a 3D tensor of 16x5x5 into 1D tensor of 16*5*5
model.add(Dense(120))                                               #-- 120 outputs fully connected layer
model.add(Activation('relu'))                                       #-- ReLU non-linearity 
model.add(Dense(84))                                                #-- 84 outputs fully connected layer
model.add(Activation('relu'))                                       #-- ReLU non-linearity 
model.add(Dense(num_classes))                                       #-- 10 outputs fully connected layer (one for each class)
model.add(Activation('softmax'))                                    #-- converts the output to a log-probability. Useful for classification problems

#print model.summary()

#-- remove # to uncomment

#model = Sequential()                                                

#-- layer 1
#model.add(Convolution2D(64, 3, 3,                                    
#                        border_mode='valid',
#                        input_shape=(3, img_rows, img_cols))) 
#model.add(Dropout(0.4))
#model.add(Convolution2D(64, 3, 3))
#model.add(Dropout(0.4))
#model.add(Activation('relu'))                                       
#model.add(MaxPooling2D(pool_size=(2, 2)))

##--layer 2
#model.add(Convolution2D(128, 3, 3))
#model.add(Dropout(0.4))                          
#model.add(Convolution2D(128, 3, 3))
#model.add(Dropout(0.4)) 
#model.add(Activation('relu'))                                       
#model.add(MaxPooling2D(pool_size=(2, 2)))

##--layer 3
#model.add(Convolution2D(256, 3, 3))
#model.add(Dropout(0.4))                          
#model.add(Convolution2D(256, 3, 3))
#model.add(Dropout(0.4)) 
#model.add(Activation('relu'))                                       
#model.add(MaxPooling2D(pool_size=(2, 2)))

##-- layer 4
#model.add(Flatten())                                                
#model.add(Dense(512))                                               
#model.add(Activation('relu'))                                       

#-- layer 5
#model.add(Dense(512))                                                
#model.add(Activation('relu'))                                       

#-- layer 6
#model.add(Dense(num_classes))                                       

#-- loss
#model.add(Activation('softmax'))

#print model.summary()


sgd = SGD(lr=learningRate, decay = lr_weight_decay)
model.compile(loss='categorical_crossentropy',
	      optimizer='sgd',
	      metrics=['accuracy'])

#-- switch verbose=0 if you get error "I/O operation from closed file"
history = model.fit(X_train, Y_train, batch_size=batchSize, nb_epoch=num_epochs,
	  verbose=0, shuffle=True, validation_data=(X_test, Y_test))



#-- summarize history for accuracy
#plt.plot(history.history['acc'])
#plt.plot(history.history['val_acc'])
#plt.title('model accuracy')
#plt.ylabel('accuracy')
#plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
#plt.show()

#-- summarize history for loss
#plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
#plt.title('model loss')
#plt.ylabel('loss')
#plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
#plt.show()

#-- test the network
score = model.evaluate(X_test, Y_test, verbose=0)


print 'acc:', history.history['acc']
print 'val_acc:', history.history['val_acc']
print 'loss:', history.history['loss']
print 'val_loss:', history.history['val_loss']
print 'Test score:', score[0] 
print 'Test accuracy:', score[1]

#cifar10_weights = model.get_weights()
#np.savez("cifar10_weights_new", cifar10_weights = cifar10_weights)












