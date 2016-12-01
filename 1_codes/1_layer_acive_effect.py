
import numpy as np
import os
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=cpu,floatX=float32"
np.random.seed(1337)  # for reproducibility

#Load the dataset
from keras.datasets import mnist
# Import the sequential module from keras
from keras.models import Sequential
# Import the layers you wish to use in your net
from keras.layers.core import Dense, Dropout, Activation
# Import the optimization algorithms that you wish to use
from keras.optimizers import SGD, Adam, RMSprop
# Import other utilities that help in data formatting etc.
from keras.utils import np_utils

batch_size = 512
nb_classes = 10
nb_epoch = 20

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Preprocess the data
# We need to shape the data into a shape that network accepts.
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)

# Here we convert the data type of data to 'float32'
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# We normalize the data to lie in the range 0 to 1.
X_train /= 255
X_test /= 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

print '##################################################################################################################################'
print 'Hidden layers lab1'
print '##################################################################################################################################'

test_loss = []
accuracy = []
batch_size = 512
hidden_layers = [16, 25, 32, 40, 50, 64, 70, 75 , 80 , 128, 256, 512, 1024, 2048]
for i in hidden_layers:
    hidden_layer = i
    print 'value is: ', i
    print
    model = Sequential()
    model.add(Dense(hidden_layer, input_shape=(784,)))
    model.add(Activation('relu'))
    model.add(Dense(hidden_layer))
    model.add(Activation('relu'))
    model.add(Dense(10))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
          metrics=['accuracy'])
    history = model.fit(X_train, Y_train,
                batch_size=batch_size, nb_epoch=nb_epoch,
                verbose=0, validation_data=(X_test, Y_test))
    score = model.evaluate(X_test, Y_test, verbose=0)
    test_loss.append(score[0])
    accuracy.append(score[1])

print "test loss: ", test_loss
print "accuray: ", accuracy

#plt.plot(accuracy)
#plt.title('model accuracy')
#plt.ylabel('accuracy')
#plt.xlabel('hidden layer size')
#plt.legend(['test'], loc='upper left')
#plt.show()
#plt.plot(test_loss)
#plt.title('model loss')
#plt.ylabel('loss')
#plt.xlabel('hidden layer size')
#plt.legend(['test'], loc='upper left')
#plt.show()

print '##################################################################################################################################'
print 'Activations'
print '##################################################################################################################################'
test_loss = []
accuracy = []
activations = ['relU','sigmoid','tanh']
for i in activations:
    print 'value is: ', i
    print
    model = Sequential()
    model.add(Dense(512, input_shape=(784,)))
    model.add(Activation(i))
    model.add(Dense(512))
    model.add(Activation(i))
    model.add(Dense(10))
    model.add(Activation(i))
    model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])
    history = model.fit(X_train, Y_train,
                    batch_size=batch_size, nb_epoch=nb_epoch,
                    verbose=0, validation_data=(X_test, Y_test))
    score = model.evaluate(X_test, Y_test, verbose=0)
    test_loss.append(score[0])
    accuracy.append(score[1])

print "test loss: ", test_loss
print "accuray: ", accuracy

