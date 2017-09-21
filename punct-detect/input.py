from nltk import FreqDist
import numpy as np
from collections import defaultdict
from tensorflow.python.framework import random_seed
from collections import namedtuple

Datasets = namedtuple('Datasets', ['train', 'test'])
Sentence = namedtuple('Sentence',['words', 'puncts'])

class PunctData():
    def __init__(self, file, cutting_freq = 10, index_dict = None, padding = True):
        self._index_dict, self._data = \
            self.read_data(file, cutting_freq, padding, index_dict)
        self.vocab_size = len(self._index_dict)
        self._id_to_word = {id: word for word,id in self._index_dict.items()}

    def generate_batch(self, raw_data, batch_size, num_steps):
        words = [sent.words for sent in raw_data]
        puncts = [sent.puncts for sent in raw_data]
        data_length = len(raw_data)
        i = 0
        # handle batch sized chunks of training data
        while i < data_length:
            start = i
            end = i + batch_size
            batch_x = np.array(words[start:end])
            batch_y = np.array(puncts[start:end])
            i = end
            j = 0
            max_size = np.max([len(word) for word in batch_x])
            divisions = max_size//num_steps
            padding_size = (divisions+1) * (num_steps)
            batch_x = [word + [0] * (padding_size-len(word)) for word in batch_x]
            batch_y = [punct + np.zeros([padding_size - len(punct),3]) if padding_size-len(punct)>0
                       else punct for punct in batch_y]
            while j < padding_size:
                x = [word[j:j+num_steps] for word in batch_x]
                y = [punct[j:j+num_steps] for punct in batch_y]
                j+= num_steps
                yield (x,y)

    def generate_epochs(self, n, batch_size, n_steps):
        for i in range(n):
            data = self.get_data()
            np.random.shuffle(data)
            yield self.generate_batch(data, batch_size, n_steps)

    def get_data(self, n_sents = 0):
        if n_sents>0:
            return self._data[:n_sents]
        return (self._data)

    def get_index_dict(self):
        return self._index_dict

    def read_data(self,file, cutting_freq, padding, index_dict):
        with open(file, encoding='utf-8') as fin:
            lines = fin.readlines()
            end_indexes = [i for (i, line) in enumerate(lines) if line == '\n']
            lines = [line.split() for line in lines]
            words = [line[0] if len(line) > 0 else None for line in lines ]
            punct = [line[1] if len(line) > 0 else None for line in lines]
            freq_dict = FreqDist(words)

            # Creating index dictionary
            if index_dict is None:
                index_dict = defaultdict(lambda: 1)
                index_dict['<PAD>'] = 0,
                index_dict['<UNK>'] = 1
                most_common = freq_dict.most_common(len(freq_dict))
                order_list = [word for (word,_) in most_common]
                for i, word in enumerate(order_list):
                    index_dict[word] = i + 2

            # Change words into indexes
            words = [word if freq_dict[word]>=cutting_freq or word is None else '<UNK>' for word in words]
            words = [index_dict[word] for word in words]

            # Change input into sentences
            sentences = []
            commas = []
            max_length = 231
            start_index = 0

            for end_index in end_indexes:
                commas.append(punct[start_index: end_index])
                sentences.append(words[start_index: end_index])
                max_length = max(max_length, end_index - start_index)
                start_index = end_index + 1

            # Padding
            # if padding:
            #     for i, sent in enumerate(sentences):
            #         if len(sent) < max_length:
            #             sent += [0] * (max_length - len(sent))
            #             commas[i] += ['O' for _ in range(max_length - len(sent))]

            data = []
            for i in range(len(commas)):
                for j,comma in enumerate(commas[i]):
                    if comma == 'COMMA':
                        commas[i][j] = [0,1,0]
                    elif comma == 'PERIOD':
                        commas[i][j] = [0,0,1]
                    else:
                        commas[i][j] = [0,0,0]
                if len(commas[i]) < max_length:
                    commas[i] += [[0,0,0] for _ in range(max_length - len(commas[i]))]
                sent = Sentence(sentences[i],commas[i])
                data.append(sent)

            return [index_dict, data]

    def id_to_word(self,id):
        return self._id_to_word[id]
    def word_to_id(self,word):
        return self._index_dict[word]
    def punct(self,id):
        if id == 0:
            return 'O'
        if id == 1:
            return ','
        return '.'


train = PunctData(r'../data/news/train.txt',10,padding=True)
test = PunctData(r'../data/news/test.txt',10,index_dict=train.get_index_dict(), padding=True)
data = Datasets(train=train,test=test)