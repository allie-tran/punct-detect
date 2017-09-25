from random import randint
from keras.models import Sequential
from keras.layers import LSTM,Dense,Activation,RepeatVector,TimeDistributed
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
model.add(LSTM(10, input_shape=(5,100),return_sequences=True)) # 2 units, 1 timesteps, 1 feature
#************************#
# model.add(RepeatVector(5))
# model.add(LSTM(10, return_sequences=True))
#************************#
model.add(TimeDistributed(Dense(100,activation='softmax'))) # fully-connected layer, outputting a prediction
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
rms = optimizers.adam(lr=0.001)
model.compile(optimizer=rms, loss = 'categorical_crossentropy',metrics=['accuracy'])

#***********************************************************************#
#***********************************************************************#
# 3. FITTING
# Once the network is compiled, it can be fit, which means adapt the weights on a training dataset.

# The network is trained using the backpropagation algorithm and optimized
# according to the optimization algorithm and loss function specified when compiling the model.

# The first layer in the network must define the number of inputs to expect.
# Input must be three-dimensional, comprised of samples, timesteps, and features.
def get_data(length, pr = False):
	X = [randint(0, 99) for _ in range(length)]
	if pr:
		print(X)
	X = indices_to_one_hot(X,100)
	res = [randint(0, 99) for _ in range(length)]
	y = indices_to_one_hot(res,100)
	# y = res
	# create X/y pairs
	# df = DataFrame(X)
	# dfy = DataFrame(y)
	# df = concat([df, dfy], axis=1)
	# df.dropna(inplace=True) #Return object with labels on given axis omitted where alternately any or all of the data are missing
	# convert to LSTM friendly format
	# values = df.values
	# X, y = values[:, 0], values[:, 1]
	X = X.reshape(-1, 5, 100) # columns are timesteps with 1 feature
	# print(X)
	# print(y)
	y = X
	# data.reshape((data.shape[0], 1, data.shape[1])) # columns are features with timestep 1
	return X,y
# configurations
BATCH_SIZE = 10

# You can reduce the amount of information displayed to just the loss each epoch by
# setting the verbose argument to 2.
# You can turn off all output by setting verbose to 1. For example:
for i in range(500):
	X, y = get_data(100)
	history = model.fit(X, y, batch_size=BATCH_SIZE, nb_epoch=1, verbose=2)
	model.reset_states()
# 4. EVALUATION
# # on new dataset
# X_eval = np.array([0.2,0.5,0.8])
# X_eval = X_eval.reshape((X_eval.shape[0],1,1))
# y_eval = np.array([0.5,0.8,0.2])
# For example, for a model compiled with the accuracy metric
loss, acc = model.evaluate(X, y)
print(loss)
# 5. MAKE PREDICTIONS
X, _ = get_data(10, pr = True)
predictions= model.predict_classes(X)

print(predictions)
# print(acc)
