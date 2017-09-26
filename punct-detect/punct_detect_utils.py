from nltk import FreqDist
from collections import defaultdict
from collections import namedtuple

import numpy as np

Sentence = namedtuple('Sentence',['words', 'puncts'])
Sentence_in_ID = namedtuple('Sentence_in_ID',['ids','p_ids'])

punct_to_id = {'O':0,'COMMA':1,'PERIOD':2}
id_to_punct = {0:'O',1:'COMMA',2:'PERIOD'}

def indices_to_one_hot(data, nb_classes):
    """Convert an iterable of indices to one-hot encoded labels."""
    targets = np.array(data).reshape(-1)
    return np.eye(nb_classes)[targets]

def read_data(file,word_to_id=None,punct_to_id=None):
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
		
		# Creating index dictionary
		if word_to_id is None:
			word_to_id = defaultdict(lambda: 1)
			word_to_id['<PAD>'] = 0,
			word_to_id['<UNK>'] = 1
			most_common = freq_dict.most_common(len(freq_dict))
			order_list = [word for (word, _) in most_common]
			for i, word in enumerate(order_list):
				word_to_id[word] = i + 2
			
		# Removing empty lines
		new_words = []
		new_puncts = []
		for i in range(len(words)):
			if words[i] == None or puncts[i] == None:
				continue
			new_words.append(words[i])
			new_puncts.append(puncts[i])
		return new_words, new_puncts, word_to_id

def process_data(words,puncts,word_to_id):
	"""Change words and punctuations into indexes"""
	ids = []
	p_ids = []
	for i in range(len(words)):
		ids.append(word_to_id[words[i]])
		p_ids.append(punct_to_id[puncts[i]])
	return ids,p_ids


words, puncts, word_to_id = read_data(r'../data/news/train.txt')
ids,p_ids = process_data(words,puncts,word_to_id)
id_to_word = {id: word for word,id in word_to_id.items()}
test_words, results, _ = read_data(r'../data/news/test.txt',word_to_id)
test_ids,test_p_ids = process_data(test_words,results,word_to_id)


def indices_to_one_hot(data, nb_classes):
	"""Convert an iterable of indices to one-hot encoded labels."""
	targets = np.array(data).reshape(-1)
	return np.eye(nb_classes)[targets]

