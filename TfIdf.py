import numpy as np
import torch
import torch.nn as nn
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

        self.idf = self.inverse_document_frequency() # inverse document frequency for each word
        self.tfidf_vectors = np.array([self.compute_tfidf_vector(self.class_data[c]) for c in self.classes]) # vector representation of tf-idf

        self.train_logistic_regression()

    def term_frequency(self, data : Sequence[str]) -> Sequence[int]:
        term_freq = np.zeros(len(self.vocab))
        frequencies = Counter(data) # gets the counts of each word in the class

        for word, count in frequencies.items():
            if word in self.vocab:
                term_freq[self.vocab[word]] = count / len(data)

        return term_freq
    
    def inverse_document_frequency(self) -> Sequence[int]:
        idf = np.zeros(len(self.vocab))

        for word in self.vocab:
            doc_freq = sum(1 for v in self.class_vocab.values() if word in v)
            idf[self.vocab[word]] = np.log(len(self.class_data) / (1 + doc_freq)) 
        return idf
    
    def compute_tfidf_vector(self, data: Sequence[str]) -> Sequence[int]:
        term_freq = self.term_frequency(data)
        
        tfidf_vector = term_freq * self.idf # multiply by IDF
        return tfidf_vector

    def train_logistic_regression(self, epochs : int = 100, loss: float = 0.01) -> None: # based off of stochastic gradient descent in Jurafsky & Martin
        self.weights = np.random.randn(len(self.classes), len(self.vocab)) # random initial weights
        self.biases = np.zeros(len(self.classes))

        x = self.tfidf_vectors.copy() # vector representation of tf-idf
        y = np.array([c for c in self.classes]) # class labels

        for epoch in range(epochs):
            # TODO: implement

            if epoch % 10 == 0:
                print("Epoch {} | Loss: {:.4}".format(epoch, loss))

    def label(self, data : Iterable[str]) -> int:
        tfidf_vector = self.compute_tfidf_vector(data)
        scores = np.dot(self.weights, tfidf_vector) + self.biases
        return np.argmax(scores)