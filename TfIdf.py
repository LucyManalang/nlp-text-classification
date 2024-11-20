import numpy as np
from typing import Sequence, Mapping, Iterable, Tuple
from collections import Counter

class TfIdf:
    def __init__(self, train_data : Iterable[Tuple[Sequence[str], int]]):
        self.classes = set([c for _, c in train_data])

        # raw data for each class
        self.class_data = {} # key = class, value = sequence of all data in class
        # vocab for each class
        self.class_vocab = {} # key = class, value = set of all words in class
        for c in self.classes:
            data = [t[0] for t in train_data if t[1] == c]
            self.class_data[c] = [item for sublist in data for item in sublist] # flatten list
            self.class_vocab[c] = set(self.class_data[c])
        
        # set of all words in the vocabulary
        self.vocab = {word: i for i, word in enumerate(set(word for v in self.class_vocab.values() for word in v))} # key = word, value = index

        self.tf = [self.term_frequency(c) for c in self.classes] # term frequency for each class
        self.idf = self.inverse_document_frequency() # inverse document frequency for each word
        self.tfidf = [[(self.tf[c][i] * self.idf[i], c, i) for i in range(len(self.vocab))] for c in self.classes]

    def term_frequency(self, c : int) -> Sequence[int]:
        term_freq = np.zeros(len(self.vocab))
        frequencies = Counter(self.class_data[c]) # gets the counts of each word in the class

        for word, count in frequencies.items():
            if word in self.vocab:
                term_freq[self.vocab[word]] = count / len(self.class_data[c])

        return term_freq
    
    def inverse_document_frequency(self) -> Sequence[int]:
        idf = np.zeros(len(self.vocab))

        for word in self.vocab:
            doc_freq = sum(1 for v in self.class_vocab.values() if word in v)
            idf[self.vocab[word]] = np.log(len(self.class_data) / (1 + doc_freq)) 
        return idf
    
    def label(self, data : Iterable[str]) -> int:
        return 0