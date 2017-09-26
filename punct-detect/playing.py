from pandas import DataFrame
from pandas import concat
from keras.models import Sequential
from keras.layers import LSTM,Dense,Activation
from keras import optimizers

import numpy as np

def indices_to_one_hot(data, nb_classes):
    """Convert an iterable of indices to one-hot encoded labels."""
    targets = np.array(data).reshape(-1)
    return np.eye(nb_classes)[targets]

#***********************************************************************#
#***********************************************************************#
# 1. DEFINING
model = Sequential()
model.add(LSTM(10, input_shape=(1,1))) # 2 units, 1 timesteps, 1 feature
model.add(Dense(3,activation='softmax')) # fully-connected layer, outputting a prediction

# The choice of activation function is most important for the output layer
# as it will define the format that predictions will take.

# Regression: Linear activation function, or ‘linear’, and the number of neurons matching the number of outputs.
# Binary Classification (2 class): Logistic activation function, or ‘sigmoid’, and one neuron the output layer.
# Multiclass Classification (>2 class): Softmax activation function, or ‘softmax’,
# and one output neuron per class value, assuming a one-hot encoded output pattern.
# model.add(Activation('softmax'))

#***********************************************************************#
#***********************************************************************#
# 2. COMPILING

# LOSS
# Regression: Mean Squared Error or ‘mean_squared_error’.
# Binary Classification (2 class): Logarithmic Loss, also called cross entropy or ‘binary_crossentropy‘.
# Multiclass Classification (>2 class): Multiclass Logarithmic Loss or ‘categorical_crossentropy‘.

# optimization
# Stochastic Gradient Descent, or ‘sgd‘, that requires the tuning of a learning rate and momentum.
# ADAM, or ‘adam‘, that requires the tuning of learning rate.
# RMSprop, or ‘rmsprop‘, that requires the tuning of learning rate.
rms = optimizers.adam(lr=0.01)
model.compile(optimizer=rms, loss = 'categorical_crossentropy',metrics=['accuracy'])

#***********************************************************************#
#***********************************************************************#
# 3. FITTING
# Once the network is compiled, it can be fit, which means adapt the weights on a training dataset.

# The network is trained using the backpropagation algorithm and optimized
# according to the optimization algorithm and loss function specified when compiling the model.

# The first layer in the network must define the number of inputs to expect.
# Input must be three-dimensional, comprised of samples, timesteps, and features.
length = 100
X = np.array([i/float(length) for i in range(length)])

res = [0] * 40 + [1] * 20 + [2] * 40
y = indices_to_one_hot(res,3)
# y = res
# create X/y pairs
# df = DataFrame(X)
# dfy = DataFrame(y)
# df = concat([df, dfy], axis=1)
# df.dropna(inplace=True) #Return object with labels on given axis omitted where alternately any or all of the data are missing
# convert to LSTM friendly format
# values = df.values
# X, y = values[:, 0], values[:, 1]
X = X.reshape(len(X), 1, 1) # columns are timesteps with 1 feature
print(X)
print(y)
# data.reshape((data.shape[0], 1, data.shape[1])) # columns are features with timestep 1

# configurations
BATCH_SIZE = 5

# You can reduce the amount of information displayed to just the loss each epoch by
# setting the verbose argument to 2.
# You can turn off all output by setting verbose to 1. For example:
history = model.fit(X, y, batch_size=BATCH_SIZE, nb_epoch=1000, verbose=2)

# 4. EVALUATION
# on new dataset
X_eval = np.array([0.2,0.5,0.8])
X_eval = X_eval.reshape((X_eval.shape[0],1,1))
y_eval = np.array([0.5,0.8,0.2])
# For example, for a model compiled with the accuracy metric
loss = model.evaluate(X, y)
print(loss)
# 5. MAKE PREDICTIONS
X_test = np.array([i/float(10) for i in range(10)])
X_test = X_test.reshape((X_test.shape[0],1,1))

predictions= model.predict_classes(X_test)

print(predictions)
# print(acc)
