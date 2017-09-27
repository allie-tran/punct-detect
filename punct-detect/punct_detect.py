from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Activation, RepeatVector, TimeDistributed, Embedding
from keras.callbacks import ModelCheckpoint
from keras import optimizers

from punct_detect_utils import *

# configurations & arguments
BATCH_SIZE = 200
VOCAB_SIZE = len(word_to_id)
PUNCT_TYPES = 3 # O, COMMA, PERIOD
TIME_STEPS = 50
TRAINING_SIZE = 149300
TESTING_SIZE = 89100
EMBEDDING_SIZE = 128
NUM_EPOCH = 200

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
	
	# 1. DEFINING THE MODEL
	model = Sequential()
	
	if not trained:
		model.add(Embedding(input_dim= VOCAB_SIZE, output_dim = EMBEDDING_SIZE))
		
		# ----------------------------------------- #
		# ------------------- NEW ----------------- #
		model.add(LSTM(128, input_shape=(TIME_STEPS, EMBEDDING_SIZE)))
		model.add(RepeatVector(TIME_STEPS))
		# now: model.output_shape == (None, TIME_STEPS, features)
		# ----------------------------------------- #
		
		model.add(LSTM(128, input_shape=(TIME_STEPS, EMBEDDING_SIZE),
		               return_sequences=True))
		model.add(TimeDistributed(Dense(PUNCT_TYPES, activation='softmax')))
		
		# 2. COMPILING
		rms = optimizers.rmsprop(lr=0.0001)
		model.compile(optimizer=rms,
		              loss='categorical_crossentropy',
		              metrics=['accuracy'])
		
		# 3. CHECKPOINTS
		checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss',
		                             verbose=1, save_best_only=True,
		                             mode='min')
		callbacks_list = [checkpoint]
		
		# 4. FITTING
		history = model.fit(X, y, validation_split= 0.33, batch_size=BATCH_SIZE,
		                    nb_epoch=NUM_EPOCH, verbose=2,
		                    callbacks=callbacks_list)
		
	# 5. LOAD BEST MODEL
	# model = load_model('best_model.h5')
	
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
			f.write("{word} {punct} {pred}\n".format(word=test_words[i],
			                                         punct=results[i],
			                                         pred=preds[i]))


if __name__ == "__main__":
	trained = input("Use trained model? (y/n):")
	run(trained=="y")