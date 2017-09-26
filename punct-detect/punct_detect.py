from random import randint
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Activation, RepeatVector, TimeDistributed
from keras import optimizers
from collections import namedtuple

import numpy as np

from punct_detect_utils import *

# configurations
BATCH_SIZE = 100
vocab_size = len(word_to_id)
punct_types = 3
time_steps = 100

load = True

def indices_to_one_hot(data, nb_classes):
	"""Convert an iterable of indices to one-hot encoded labels."""
	targets = np.array(data).reshape(-1)
	return np.eye(nb_classes)[targets]

# Input must be three-dimensional, comprised of samples, timesteps, and features.
def get_data(train_or_test="train"):
	if train_or_test == "train":
		X = indices_to_one_hot(ids, vocab_size)[:149300]
		y = indices_to_one_hot(p_ids, punct_types)[:149300]
	else:
		X = indices_to_one_hot(test_ids, vocab_size)[:89100]
		y = indices_to_one_hot(test_p_ids, punct_types)[:89100]
	
	X = X.reshape(-1, time_steps, vocab_size)  # columns are timesteps with 1 feature
	y = y.reshape(-1, time_steps, punct_types)
	# print(X)
	# print(y)
	return X, y

X, y = get_data("train")

model = Sequential()

if load == False:
	# ***********************************************************************#
	# ***********************************************************************#
	# 1. DEFINING
	model.add(LSTM(100, input_shape=(time_steps, vocab_size), return_sequences=True))
	model.add(
		TimeDistributed(Dense(punct_types, activation='softmax')))  # fully-connected layer, outputting a prediction
	# The choice of activation function is most important for the output layer
	# as it will define the format that predictions will take.
	
	# Regression: Linear activation function, or ‘linear’, and the number of neurons matching the number of outputs.
	# Binary Classification (2 class): Logistic activation function, or ‘sigmoid’, and one neuron the output layer.
	# Multiclass Classification (>2 class): Softmax activation function, or ‘softmax’,
	# and one output neuron per class value, assuming a one-hot encoded output pattern.
	# model.add(Activation('softmax'))
	
	# ***********************************************************************#
	# ***********************************************************************#
	# 2. COMPILING
	
	# LOSS
	# Regression: Mean Squared Error or ‘mean_squared_error’.
	# Binary Classification (2 class): Logarithmic Loss, also called cross entropy or ‘binary_crossentropy‘.
	# Multiclass Classification (>2 class): Multiclass Logarithmic Loss or ‘categorical_crossentropy‘.
	
	# optimization
	# Stochastic Gradient Descent, or ‘sgd‘, that requires the tuning of a learning rate and momentum.
	# ADAM, or ‘adam‘, that requires the tuning of learning rate.
	# RMSprop, or ‘rmsprop‘, that requires the tuning of learning rate.
	rms = optimizers.rmsprop(lr=0.001)
	model.compile(optimizer=rms, loss='categorical_crossentropy', metrics=['accuracy'])
	
	# ***********************************************************************#
	# ***********************************************************************#
	# 3. FITTING
	# Once the network is compiled, it can be fit, which means adapt the weights on a training dataset.
	
	# The network is trained using the backpropagation algorithm and optimized
	# according to the optimization algorithm and loss function specified when compiling the model.
	
	# The first layer in the network must define the number of inputs to expect.
	
	# You can reduce the amount of information displayed to just the loss each epoch by
	# setting the verbose argument to 2.
	# You can turn off all output by setting verbose to 1. For example:
	history = model.fit(X, y, batch_size=BATCH_SIZE, nb_epoch=100, verbose=2)
	
	model.save('finished_model.h5')
else:
	model = load_model('finished_model.h5')
# 4. EVALUATION
# # on new dataset
# X_eval = np.array([0.2,0.5,0.8])
# X_eval = X_eval.reshape((X_eval.shape[0],1,1))
# y_eval = np.array([0.5,0.8,0.2])
# For example, for a model compiled with the accuracy metric
# loss, acc = model.evaluate(X, y)
# print(loss)
# 5. MAKE PREDICTIONS
print('Predicting...')
with open('train_result.txt', 'w',encoding='utf_8') as f:
	predictions = model.predict_classes(X)
	predictions = predictions.reshape(-1)
	preds = [id_to_punct[p_ids] for p_ids in predictions]
	for i in range(149300):
		f.write("{word} {punct} {pred}\n".format(word=words[i],punct=puncts[i], pred=preds[i]))

with open('test_result.txt', 'w',encoding='utf_8') as f:
	X, y = get_data(train_or_test="test")
	predictions = model.predict_classes(X)
	predictions = predictions.reshape(-1)
	preds = [id_to_punct[p_ids] for p_ids in predictions]
	for i in range(89100):
		f.write("{word} {punct} {pred}\n".format(word=test_words[i], punct=results[i], pred=preds[i]))
		# print(acc)
