import numpy as np
import os
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=cpu,floatX=float32"
import theano
import keras
from PIL import Image
from keras.preprocessing import image as image_utils
from keras.models  import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.utils import np_utils
import matplotlib.pyplot as plt
from keras import backend as K
K.set_image_dim_ordering('th')

pubfig = np.load('data/lab2/lab2_data/pubfig_data.npz')
X_train_face = pubfig['X_train_face']
y_train_face = pubfig['y_train_face']
X_test_face = pubfig['X_test_face']
y_test_face = pubfig['y_test_face']


print "Training data:"
print "Number of examples:", X_train_face.shape[0]
print "Number of channels:", X_train_face.shape[1]
print "Image size:", X_train_face.shape[2], X_train_face.shape[3]

print

print "Test data:"
print "Number of examples:", X_test_face.shape[0]
print "Number of channels:", X_test_face.shape[1]
print "Image size:", X_test_face.shape[2], X_test_face.shape[3]

plot = []
for i in range(1,10):
    plot_image = X_train_face[100*i,:,:,:].transpose(1,2,0)
    for j in range(1,10):
        plot_image = np.concatenate((plot_image, X_train_face[100*i+j,:,:,:].transpose(1,2,0)), axis=1)
    if i==1:
        plot = plot_image
    else:
        plot = np.append(plot, plot_image, axis=0)

#plt.imshow(plot)
#plt.axis('off')
#plt.show()





def VGG_16(weights_path=None):
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))

    if weights_path:
        model.load_weights(weights_path)

    return model

num_face_classes = 60                    #-- number of classes in pubfig dataset
num_cifar_classes = 10                   #-- number of classes in CIFAR-10 dataset

img_rows, img_cols = 32, 32              #-- input image dimensions

model = VGG_16('vgg16_weights.h5')

model.layers.pop()
model.layers.pop()
model.outputs = [model.layers[-1].output]
model.layers[-1].outbound_nodes = []


model.add(Dense(num_face_classes))
model.add(Activation('softmax'))

print "mean before normalization:", np.mean(X_train_face) 
print "std before normalization:", np.std(X_train_face)


mean=[0,0,0]
std=[0,0,0]
newX_train_face = np.ones(X_train_face.shape)
newX_test_face = np.ones(X_test_face.shape)
for i in xrange(3):
    mean[i] = np.mean(X_train_face[:,i,:,:])
    std[i] = np.std(X_train_face[:,i,:,:])
    
for i in xrange(3):
    newX_train_face[:,i,:,:] = X_train_face[:,i,:,:] - mean[i]
    newX_train_face[:,i,:,:] = newX_train_face[:,i,:,:] / std[i]
    newX_test_face[:,i,:,:] = X_test_face[:,i,:,:] - mean[i]
    newX_test_face[:,i,:,:] = newX_test_face[:,i,:,:] / std[i]
        
    
X_train_face = newX_train_face
X_test_face = newX_test_face

print "mean after normalization:", np.mean(X_train_face)
print "std after normalization:", np.std(X_train_face)

batchSize = 50                    #-- Training Batch Size
num_epochs = 10                   #-- Number of epochs for training   
learningRate= 0.001               #-- Learning rate for the network
lr_weight_decay = 0.99            #-- Learning weight decay. Reduce the learn rate by 0.99 after epoch

Y_train = np_utils.to_categorical(y_train_face-1, num_face_classes)
Y_test = np_utils.to_categorical(y_test_face-1, num_face_classes)

fine_tune_learningRate=0.00001
sgd = SGD(lr=learningRate, decay = lr_weight_decay)
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
print type(X_train_face[0])
print type(X_train_face[0][0])
print type(X_train_face[0][0])

npad = ((0,0), (0,0), (96,96), (96,96))
X_train_face = X_train_face[:30]
Y_train = Y_train[:30]
X_test_face = X_test_face[:3]
Y_test = Y_test[:3]

b = np.pad(X_train_face, pad_width=npad, mode='constant', constant_values=0)
X_train_face = b
b = np.pad(X_test_face, pad_width=npad, mode='constant', constant_values=0)
X_test_face = b

print 'padding done'
history = model.fit(X_train_face, Y_train, batch_size=batchSize, nb_epoch=num_epochs,
          verbose=1, shuffle=True, validation_data=(X_test_face, Y_test))


score = model.evaluate(X_test_face, Y_test, verbose=0)

print 'loss: ', history.history['loss']
print 'val_loss:', history.history['val_loss']
print 'acc: ', history.history['acc']
print 'val_acc:', history.history['val_acc']
print 'Test score:', score[0] 
print 'Test accuracy:', score[1]
