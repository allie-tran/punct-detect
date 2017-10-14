from keras.models import Sequential, load_model, Model, Input
from keras.layers import LSTM, Dense, Activation, RepeatVector, TimeDistributed, Embedding, Bidirectional, Dropout
from keras.layers import Flatten, Permute, Lambda, Merge, Reshape
from keras.layers.merge import Multiply,multiply
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import optimizers
from keras.regularizers import l2 # L2-regularisation
import keras.backend as K
from sklearn.utils import class_weight
from keras.utils import plot_model
import matplotlib.pyplot as plt
import os
import functools
from Fscore import statistic

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'


from punct_detect_utils import *
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)


# configurations & arguments
BATCH_SIZE = 16
VOCAB_SIZE = len(word_to_id)
PUNCT_TYPES = len(punct_to_id) # O, COMMA, PERIOD
TIME_STEPS = max_length
# TRAINING_SIZE = (TIME_STEPS*BATCH_SIZE) * (len(ids)//(TIME_STEPS*BATCH_SIZE))
# TESTING_SIZE = (TIME_STEPS*BATCH_SIZE) * (len(test_ids)//(TIME_STEPS*BATCH_SIZE))
EMBEDDING_SIZE = 128
HIDDEN = 256
NUM_EPOCH = 50
LEARNING_RATE = 0.001

# Input must be 3D, comprised of samples, timesteps, and features.
def get_data_batch_1(l, ids, p_ids):
    """Get data suitable for the model"""
    X = np.reshape(ids,(-1,l))
    y = [[indices_to_one_hot(p_id, PUNCT_TYPES) for p_id in pids]
         for pids in p_ids]
    y = np.reshape(y,(-1,l,PUNCT_TYPES))
    return X, y


# *************************************************************************** #
# ******************************* BEGIN HERE ******************************** #
# *************************************************************************** #

def run(trained = False):
    
    # 1. DEFINING THE MODE
    a = Input(batch_shape=(1,None))
    embedding = Embedding(input_dim=VOCAB_SIZE,output_dim=EMBEDDING_SIZE)(a)
    # ----------------------------------------- #
    dropout = Dropout(0.5)(embedding)
    attention = Dense(EMBEDDING_SIZE,activation='sigmoid')(dropout)
    new_input = Multiply()([embedding,attention])
    # ----------------------------------------- #
    lstm1 = LSTM(HIDDEN, return_sequences=True,
                                 kernel_initializer='lecun_uniform')(new_input)
    dropout1 = Dropout(0.5)(lstm1)
    lstm2 = LSTM(HIDDEN, return_sequences=True,
                   kernel_initializer='lecun_uniform',
                   go_backwards=True)(dropout1)
    # (batch, step, hidden)
    # # -----------------------------------------#
    dropout2 = Dropout(0.5)(lstm2)
    lstm3 = LSTM(HIDDEN, return_sequences=True,
                 kernel_initializer='lecun_uniform')(dropout2)
    dropout3 = Dropout(0.5)(lstm3)
    lstm4 = LSTM(HIDDEN, return_sequences=True,
                 kernel_initializer='lecun_uniform',
                 go_backwards=True)(dropout3)
    # (batch, step, hidden)
    # ------------------------------------------#
    dropout4 = Dropout(0.5)(lstm4)
    final = TimeDistributed(Dense(PUNCT_TYPES, activation='sigmoid'))(dropout4)
    model = Model(inputs=a,outputs=final)
    # (batch, step, punct_types)
    # 2. COMPILING
    opt = optimizers.rmsprop(lr=LEARNING_RATE)
    model.compile(optimizer=opt,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    # 3. CHECKPOINTS
    # checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss',
    #                              verbose=1, save_best_only=True,
    #                              mode='min',save_weights_only=True)
    # earlyStopping = EarlyStopping(monitor='val_loss', patience=3, verbose=0, mode='min', min_delta=1e-5)
    # callbacks_list = [earlyStopping, checkpoint]
    
    if not trained:
        # # 4. FITTING
        plot_model(model, to_file='model.png')
        model.summary()
        # history = model.fit(X, y, validation_split= 0.2, batch_size=BATCH_SIZE,
        #                     epochs=NUM_EPOCH, verbose=2, class_weight=d_weight,
        #                     callbacks=callbacks_list)
        # fit network
        for i in range(NUM_EPOCH):
            print("=================================================")
            print("EPOCH", i)
            print("=========")
            for l in words.keys():
                X, y = get_data_batch_1(l, ids[l], p_ids[l])
                # print('-------------------------------------------')
                # print(X)
                # print(y)
                model.fit(X,y, class_weight=[0,10,10,20,1],
                          epochs=1, batch_size=1, verbose=0, shuffle=False)
            
            showid = np.random.randint(20)
            while showid not in ids:
                showid = np.random.randint(20)
            predictions = model.predict(np.reshape(ids[showid][0],(1,-1)), batch_size=1)
            predictions = np.argmax(predictions, axis=2)
            print(predictions.shape)
            predictions = predictions.reshape(-1)
            # change from id to punctuations
            preds = [id_to_punct[p_ids] for p_ids in predictions]
            print(words[showid][0])
            print(puncts[showid][0])
            print(preds)
            if i % 5 == 0:
                model.save_weights("model_iter_{iter}.h5".format(iter=i))

        model.save_weights("final_model.h5")
        
    # 5. LOAD BEST MODEL
    model.load_weights('model_iter_20.h5')
    
    # 6. EVALUATION
    # loss, acc = model.evaluate(X, y)
    
    # 7. MAKE PREDICTIONS
    print('Predicting...')
    # On training data
    with open('../result/train_result.txt', 'w',encoding='utf_8') as f:
        for l in words.keys():
            X, y = get_data_batch_1(l, ids[l], p_ids[l])
            for j in range(len(X)):
                predictions = model.predict(np.reshape(X[j],(1,-1)))
                predictions = np.argmax(predictions, axis=2)
                # print(predictions.shape)
                predictions = predictions.reshape(-1)
                # change from id to punctuations
                preds = [id_to_punct[p_ids] for p_ids in predictions]
                for k in range(l):
                    f.write("{word} {punct} {pred}\n".format(word=words[l][j][k],
                                                         punct=id_to_punct[p_ids[l][j][k]],
                                                         pred=preds[k]))

    # On testing data
    with open('../result/test_result.txt', 'w',encoding='utf_8') as f:
        for l in words.keys():

            X, y = get_data_batch_1(l, test_ids[l], test_p_ids[l])
            for j in range(len(X)):
                predictions = model.predict(np.reshape(X[j],(1,-1)))
                predictions = np.argmax(predictions, axis=2)
                # print(predictions.shape)
                predictions = predictions.reshape(-1)
                # change from id to punctuations
                preds = [id_to_punct[p_ids] for p_ids in predictions]
                for k in range(l):
                    f.write("{word} {punct} {pred}\n".format(word=test_words[l][j][k],
                                                         punct=id_to_punct[test_p_ids[l][j][k]],
                                                         pred=preds[k]))



if __name__ == "__main__":
    trained = input("Use trained model? (y/n):")
    statistic('../result/test_result.txt')
    statistic('../result/train_result.txt')