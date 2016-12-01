from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
import numpy as np 

X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])

model = Sequential()
model.add(Dense(8, input_dim=2))
model.add(Activation('tanh'))
model.add(Dense(1))
model.add(Activation('sigmoid'))

sgd = SGD(lr=0.1)
model.compile(loss='binary_crossentropy', optimizer=sgd)

model.fit(X, y, verbose=0,show_accuracy=True, batch_size=1, nb_epoch=1000)
x = np.random.random_sample((70000,))
y = np.random.random_sample((70000,))
t = []
for i in range(len(x)):
      t.append([x[i], y[i]])  

z = model.predict_classes(t)
for i in range(len(x)):
    if(z[i][0] < 0.5):
            plt.plot(x[i], y[i], 'b.')
    else:
            plt.plot(x[i], y[i], 'r.')
plt.show()
