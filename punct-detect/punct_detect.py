from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Activation, RepeatVector, TimeDistributed, Embedding, Bidirectional, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import optimizers
from sklearn.utils import class_weight



from punct_detect_utils import *

# configurations & arguments
BATCH_SIZE = 8
VOCAB_SIZE = len(word_to_id)
PUNCT_TYPES = len(punct_to_id) # O, COMMA, PERIOD
TIME_STEPS = max_length
TRAINING_SIZE = (len(ids)//(TIME_STEPS*BATCH_SIZE)) * (TIME_STEPS*BATCH_SIZE)
TESTING_SIZE = (len(test_ids)//(TIME_STEPS*BATCH_SIZE)) * (TIME_STEPS*BATCH_SIZE)
EMBEDDING_SIZE = 128
NUM_EPOCH = 50
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
	
	# 1. DEFINING THE MODEL
	model = Sequential()
	
	if not trained:
		model.add(Embedding(input_dim= VOCAB_SIZE, output_dim = EMBEDDING_SIZE))
		
		# ----------------------------------------- #
		# ------------------- NEW ----------------- #
		# model.add(LSTM(128, input_shape=(TIME_STEPS, EMBEDDING_SIZE)))
		# model.add(RepeatVector(TIME_STEPS))
		# now: model.output_shape == (None, TIME_STEPS, features)
		# ----------------------------------------- #
		
		model.add(Bidirectional(LSTM(128, input_shape=(TIME_STEPS, EMBEDDING_SIZE),
		               return_sequences=True)))
		model.add(Activation('softmax'))
		model.add(Dropout(0.2))
		model.add(TimeDistributed(Dense(PUNCT_TYPES, activation='softmax')))
		
		# 2. COMPILING
		opt = optimizers.adam(lr=LEARNING_RATE)
		model.compile(optimizer=opt,
		              loss='categorical_crossentropy',
		              metrics=['accuracy'])
		
		# 3. CHECKPOINTS
		checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss',
		                             verbose=1, save_best_only=True,
		                             mode='min')
		earlyStopping = EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='min',min_delta=1e-5)
		callbacks_list = [checkpoint, earlyStopping]
		# 4. FITTING
		model.summary()
		history = model.fit(X, y, validation_split= 0.2, batch_size=BATCH_SIZE,
		                    epochs=NUM_EPOCH, verbose=2,
		                    callbacks=callbacks_list, class_weight=weight)
		model.save("final_model.h5")
		
	# 5. LOAD BEST MODEL
	model = load_model('best_model.h5')
	
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