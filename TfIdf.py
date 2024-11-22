import math
import torch
from typing import Sequence, Iterable, Tuple
from collections import Counter
from logistic_regression import LogisticRegression

from sklearn.feature_extraction.text import TfidfVectorizer


class TfIdf:
    def __init__(self, train_data : Iterable[Tuple[Sequence[str], int]], scikit_learn : bool = False):
        self.skikit_learn = scikit_learn
        self.classes = set([c for _, c in train_data])
        self.class_data = {} # raw data for each class | key = class, value = sequence of all data in class
        self.class_vocab = {} # vocab for each class | key = class, value = set of all words in class

        for c in self.classes:
            data = [t[0] for t in train_data if t[1] == c]
            self.class_data[c] = [item for sublist in data for item in sublist] # flatten list
            self.class_vocab[c] = set(self.class_data[c])

        # from https://scikit-learn.org/1.5/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
        # we used this to test the performance of our model compared to a pre-implemented one, this is NOT our submission
        if scikit_learn:
            corpus = [" ".join(data) for data in self.class_data.values()]

            # removes very common words to improve performance
            self.vectorizer = TfidfVectorizer(stop_words='english')

            tfidf_vectors = torch.tensor(self.vectorizer.fit_transform(corpus).toarray(), dtype=torch.float)
            self.vocab = self.vectorizer.get_feature_names_out()
        else:
            # all enumerated words in the vocabulary
            self.vocab = {word: i for i, word in enumerate(set(word for v in self.class_vocab.values() for word in v))} # key = word, value = index

            self.idf = self.inverse_document_frequency()
            tfidf_vectors = torch.stack([self.compute_tfidf_vector(self.class_data[c]) for c in self.classes]) 

        # logistic regression model
        self.model = LogisticRegression(len(self.vocab), len(self.classes))
        self.model.train_model(tfidf_vectors, torch.tensor([c for c in self.classes], dtype=torch.long))

    def term_frequency(self, data : Sequence[str]) -> torch.Tensor:
        term_freq = torch.zeros(len(self.vocab))
        frequencies = Counter(data) # gets the counts of each word in the class
        max_freq = max(frequencies.values()) # normalizes the frequencies

        for word, count in frequencies.items():
            if word in self.vocab:
                term_freq[self.vocab[word]] = (1 + math.log(count)) / max_freq

        return term_freq
    
    def inverse_document_frequency(self) -> torch.Tensor:
        idf = torch.zeros(len(self.vocab))
        epsilon = 1e-6 # avoids log(0)

        for word in self.vocab:
            doc_freq = sum(1 for v in self.class_vocab.values() if word in v)
            idf[self.vocab[word]] = math.log((len(self.class_data) + epsilon) / (1 + doc_freq))
        return idf
    
    def compute_tfidf_vector(self, data: Sequence[str]) -> torch.Tensor:
        term_freq = self.term_frequency(data)
        
        tfidf_vector = term_freq * self.idf # multiply by IDF
        return tfidf_vector

    def label(self, data : Iterable[str]) -> int:
        self.model.eval()  
        with torch.no_grad():
            if self.skikit_learn:
                tfidf_vector = torch.tensor(self.vectorizer.transform([" ".join(data)]).toarray(), dtype=torch.float)
                logits = self.model(tfidf_vector)
            else:
                tfidf_vector = self.compute_tfidf_vector(data)
                logits = self.model(tfidf_vector.unsqueeze(0))
            predicted_class = torch.argmax(logits, dim=1).item()
        return predicted_class