from keras.models import Sequential, load_model, Model, Input
from keras.layers import LSTM, Dense, Activation, RepeatVector, TimeDistributed, Embedding, Bidirectional, Dropout
from keras.layers import Flatten, Permute, Lambda, Merge, Reshape
from keras.layers.merge import Multiply,multiply
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import optimizers
import keras.backend as K
from sklearn.utils import class_weight
from keras.utils import plot_model


from punct_detect_utils import *
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)


# configurations & arguments
BATCH_SIZE = 32
VOCAB_SIZE = len(word_to_id)
PUNCT_TYPES = len(punct_to_id) # O, COMMA, PERIOD
TIME_STEPS = 81
TRAINING_SIZE = (TIME_STEPS*BATCH_SIZE) * (len(ids)//(TIME_STEPS*BATCH_SIZE))
TESTING_SIZE = (TIME_STEPS*BATCH_SIZE) * (len(test_ids)//(TIME_STEPS*BATCH_SIZE))
EMBEDDING_SIZE = 64
HIDDEN = 64
NUM_EPOCH = 500
LEARNING_RATE = 0.001

# Input must be 3D, comprised of samples, timesteps, and features.
def get_data(train_or_test="train"):
	"""Get data suitable for the model"""
	if train_or_test == "train":
		X = np.array(ids)[:TRAINING_SIZE]
		# 	X = indices_to_one_hot(ids, vocab_size)[:149300]
		y = indices_to_one_hot(p_ids, PUNCT_TYPES)[:TRAINING_SIZE]
	else:
		X = np.array(test_ids)[:TESTING_SIZE]
		# 	X = indices_to_one_hot(test_ids, vocab_size)[:89100]
		y = indices_to_one_hot(test_p_ids, PUNCT_TYPES)[:TESTING_SIZE]
	
	X = X.reshape(-1, TIME_STEPS)  # columns are timesteps with 1 feature
	y = y.reshape(-1, TIME_STEPS, PUNCT_TYPES)

	return X, y



# *************************************************************************** #
# ******************************* BEGIN HERE ******************************** #
# *************************************************************************** #

def run(trained = False):
	X, y = get_data("train")
	weight = class_weight.compute_class_weight('balanced',
	                                           np.unique(p_ids[:TRAINING_SIZE]),
	                                           p_ids[:TRAINING_SIZE])
	unique = np.unique(p_ids)
	d_weights = np.ones(len(unique))
	for i in range(len(unique)):
		d_weights[punct_to_id[unique[i]]] = weight[i]
	weight = d_weights
	# 1. DEFINING THE MODEL
	# 1.1 The LSTM  model -  output_shape = (batch, step, hidden)
	lstm = Sequential()
	lstm.add(Embedding(input_dim=VOCAB_SIZE, input_length=TIME_STEPS,
	                   output_dim=EMBEDDING_SIZE))
	
	# ----------------------------------------- #
	# ------------------- NEW ----------------- #
	# model.add(LSTM(128, input_shape=(TIME_STEPS, EMBEDDING_SIZE)))
	# model.add(RepeatVector(TIME_STEPS))
	# now: model.output_shape == (None, TIME_STEPS, features)
	# ----------------------------------------- #
	history = Bidirectional(LSTM(HIDDEN, return_sequences=True,
	                             kernel_initializer='lecun_uniform'))
	lstm.add(history)
	# lstm.add(Reshape((TIME_STEPS,HIDDEN)))
	# lstm.add(Dense(HIDDEN, activation="softmax"))
	# -----------------------------------------#
	# -----------------------------------------#
	# 1.2 The attention model
	# actual output shape  = (batch, step)
	# after reshape : output_shape = (batch, step, hidden)
	attention = Sequential()
	attention.add(lstm)
	attention.add(Dense(1, input_shape=(TIME_STEPS, VOCAB_SIZE),
	                    activation='tanh',kernel_initializer='he_uniform'))
	attention.add(Flatten())
	attention.add(Activation('softmax'))
	attention.add(RepeatVector(2 * HIDDEN))
	att = Permute([2, 1])
	attention.add(att)
	
	# 1.3 Final model
	model = Sequential()
	model.add(Merge([history, att], mode='mul'))
	# model.add(TimeDistributed(Lambda(lambda xin: K.sum(xin, axis=1))))
	
	# -----------------------------------------#
	# -----------------------------------------#
	# model.add(RepeatVector(1))
	model.add(Dropout(0.2))
	model.add(TimeDistributed(Dense(PUNCT_TYPES, activation='softmax')))
	
	# 2. COMPILING
	
	opt = optimizers.rmsprop(lr=LEARNING_RATE)
	model.compile(optimizer=opt,
	              loss='categorical_crossentropy',
	              metrics=['categorical_accuracy'])
	
	# 3. CHECKPOINTS
	checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss',
	                             verbose=1, save_best_only=True,
	                             mode='min',save_weights_only=True)
	earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min', min_delta=1e-5)
	callbacks_list = [earlyStopping, checkpoint]
	
	if not trained:
		# 4. FITTING
		lstm.summary()
		attention.summary()
		model.summary()
		plot_model(model, to_file='model.png')
		model.fit(X, y, validation_split= 0.2, batch_size=BATCH_SIZE,
		                    epochs=NUM_EPOCH, verbose=2,
		                    callbacks=callbacks_list, class_weight=weight)
		model.save_weights("final_model.h5")
		
	# 5. LOAD BEST MODEL
	model.load_weights('best_model.h5')
	
	# 6. EVALUATION
	# loss, acc = model.evaluate(X, y)
	
	# 7. MAKE PREDICTIONS
	print('Predicting...')
	# On training data
	with open('../result/train_result.txt', 'w',encoding='utf_8') as f:
		predictions = model.predict_classes(X)
		predictions = predictions.reshape(-1) # flatten
		# change from id to punctuations
		preds = [id_to_punct[p_ids] for p_ids in predictions]
		for i in range(TRAINING_SIZE):
			if words[i] != '<PAD>':
				f.write("{word} {punct} {pred}\n".format(word=words[i],
			                                         punct=puncts[i],
			                                         pred=preds[i]))
	
	# On testing data
	with open('../result/test_result.txt', 'w',encoding='utf_8') as f:
		X, y = get_data(train_or_test="test")
		predictions = model.predict_classes(X)
		predictions = predictions.reshape(-1)
		preds = [id_to_punct[p_ids] for p_ids in predictions]
		for i in range(TESTING_SIZE):
			if test_words[i] != '<PAD>':
				f.write("{word} {punct} {pred}\n".format(word=test_words[i],
			                                         punct=results[i],
			                                         pred=preds[i]))


if __name__ == "__main__":
	# trained = input("Use trained model? (y/n):")
	run(False)