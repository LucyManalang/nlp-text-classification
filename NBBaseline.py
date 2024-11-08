import numpy as np
from typing import Sequence, Mapping, Iterable
from data import *

class NBBaseline:
    def __init__(self, train_data : Iterable[tuple[Sequence[str], int]]):
        self.classes = set([c for _, c in train_data])

        self.vocab = set()
        self.class_data = {}
        for t in train_data:
            self.vocab.update(t[0])

        self.vocab_size = len(self.vocab)

        self.ulm = {}
        for c in self.classes:
            ulm_class = [t[0] for t in train_data if t[1] == c]
            self.ulm[c] = self.get_logfreqs([item for sublist in ulm_class for item in sublist])

    def get_logfreqs(self, data : Iterable[str]) -> Mapping[str, int]: 
        counts = {}
        total = 0
        for w in data:
            counts[w] = counts.get(w, 0) + 1
            total += 1
        
        logprobs = {}
        smoothed_total = total * self.vocab_size
        for w in self.vocab:
            smoothed_count = counts.get(w, 0) + 1
            logprobs[w] = np.log(smoothed_count / smoothed_total)
        
        return logprobs

    def label(self, data : Iterable[str]) -> int:
        log_likelihoods = {}
        for c in self.classes:
            log_likelihood = 0
            for w in data:
                log_likelihood += self.ulm[c].get(w)
            log_likelihoods[c] = log_likelihood

        return max(log_likelihoods, key=log_likelihoods.get)
