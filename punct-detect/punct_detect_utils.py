from nltk import FreqDist
from collections import defaultdict
from collections import namedtuple

import numpy as np

Sentence = namedtuple('Sentence',['words', 'puncts'])
Sentence_in_ID = namedtuple('Sentence_in_ID',['ids','p_ids'])

def indices_to_one_hot(data, nb_classes):
    """Convert an iterable of indices to one-hot encoded labels."""
    targets = np.array(data).reshape(-1)
    return np.eye(nb_classes)[targets]

def read_data(file, cutting_freq = 5,
              word_to_id=None, id_to_word = None,
              punct_to_id=None, id_to_punct=None):
	"""Return:
	data: an array of (sentence, puncts)
	id_to_word: an index-to-word dictionary of words in all sentences
	word_to_id: a word-to-index dictionary of words
	id_to_punct: an index-to-punctuation dictionary
	punct_to_id: an punctuation-to-id dictionary
	"""
	with open(file, encoding='utf-8') as fin:
		lines = fin.readlines()
		end_indexes = [i for (i, line) in enumerate(lines) if line == '\n']
		lines = [line.split() for line in lines]
		words = [line[0] if len(line) > 0 else None for line in lines]
		puncts = [line[1] if len(line) > 0 else None for line in lines]
		freq_dict = FreqDist(words)
		
		# Removing empty lines
		new_words = []
		new_puncts = []
		for i in range(len(words)):
			if words[i] == None or puncts[i] == None:
				continue
			new_words.append(words[i])
			new_puncts.append(puncts[i])
		
		words = new_words
		puncts = new_puncts
		
		# Creating index dictionary for words
		if word_to_id is None:
			word_to_id = defaultdict(lambda: 1)
			word_to_id['<PAD>'] = 0
			word_to_id['<UNK>'] = 1
			id_to_word = defaultdict(lambda: '<UNK>')
			most_common = freq_dict.most_common(len(freq_dict))
			order_list = [word for (word, freq) in most_common
			              if freq>cutting_freq]
			for i, word in enumerate(order_list):
				word_to_id[word] = i + 2
				id_to_word[i + 2] = word
				
		# Creating index dictionary for punctuations
		if punct_to_id is None:
			punct_to_id = defaultdict(lambda: 0)
			punct_list = np.unique(puncts+['O'])
			id_to_punct = {}
			for i, punct in enumerate(punct_list):
				punct_to_id[punct] = i
				id_to_punct[i] = punct
		
		# Padding
		pad_words = []
		pad_puncts = []
		max_length = 86
		start_index = 0
		for end_index in end_indexes:
			max_length = max(max_length,end_index-start_index+1)
			start_index = end_index + 1
			
		start_index = 0
		for end_index in end_indexes:
			pad_puncts = pad_puncts + puncts[start_index: end_index]
			pad_words = pad_words + words[start_index:end_index]
			padding = max_length - (end_index-start_index+1)
			if padding>0:
				pad_words += ['<PAD>'] * padding
				pad_puncts += ['O'] * padding
			start_index = end_index + 1
		
		return pad_words, pad_puncts, word_to_id, id_to_word, punct_to_id, id_to_punct, max_length

def process_data(words,puncts,word_to_id,punct_to_id):
	"""Change words and punctuations into indexes"""
	ids = []
	p_ids = []
	for i in range(len(words)):
		ids.append(word_to_id[words[i]])
		p_ids.append(punct_to_id[puncts[i]])
	return ids,p_ids


words, puncts, word_to_id, id_to_word, punct_to_id, id_to_punct, max1 = \
	read_data(r'../data/run/punc/punc.tr',5)
ids,p_ids = process_data(words,puncts,word_to_id,punct_to_id)

test_words, results, _ , _ , _, _, max2= \
	read_data(r'../data/run/punc/punc.ts',5,
	          word_to_id,id_to_word,punct_to_id,id_to_punct)
test_ids,test_p_ids = process_data(test_words,results,word_to_id,punct_to_id)
max_length = max(max1,max2)