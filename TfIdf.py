import numpy as np
from typing import Sequence, Mapping, Iterable, Tuple
from collections import Counter

class TfIdf:
    def __init__(self, train_data : Iterable[Tuple[Sequence[str], int]]):
        self.classes = set([c for _, c in train_data])

        self.class_data = {} # key = class, value = sequence of all words in class
        self.vocab = {} # key = word, value = index
        for c in self.classes:
            data = [t[0] for t in train_data if t[1] == c]
            self.class_data[c] = [item for sublist in data for item in sublist]
        self.vocab = {word: i for i, word in enumerate(set(self.class_data[c] for c in self.classes))}

        idf = self.inverse_document_frequency()
        # self.tfidf_vectors = np.array([self.term_frequency(c) * idf for c in self.classes])


        
    def term_frequency(self, c : int) -> Sequence[int]:
        term_freq = np.zeros(len(self.vocab))
        frequencies = Counter(self.class_data[c])

        for word, count in frequencies.items():
            if word in self.vocab:
                term_freq[self.vocab[word]] = count
        
        return term_freq
    
    def inverse_document_frequency(self) -> Sequence[int]:
        idf = np.zeros(len(self.vocab))

        for word in self.vocab:
            doc_freq = 0
            for data in self.class_data.values():
                if word in data:
                    doc_freq += 1
            idf[self.vocab[word]] = np.log(len(self.class_data) / (doc_freq + 1))
        return idf
    
    def label(self, data : Iterable[str]) -> int:
        print(self.tfidf_vectors)
        return np.argmax(np.dot(self.tfidf_vectors, np.array([self.term_frequency(self.class_vocab[c], c) for c in self.classes])))