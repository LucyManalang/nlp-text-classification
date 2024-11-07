import numpy as np
from typing import Sequence, Mapping, Iterable

class NBBaseline:
    def __init__(self, train_data : Mapping[Sequence[str], str]):
        self.classes = set(train_data.values())
        self.ulm = train_data
        
        self.vocab = set(train_data.keys())
        
        self.vocab_size = len(self.vocab)

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

    def label(self, data : Iterable[str]) -> str:
        log_likelihoods = {}
        for c in self.classes:
            log_likelihood = 0
            for w in data:
                log_likelihood += self.ulm[c].get(w)
            log_likelihoods[c] = log_likelihood

        return max(log_likelihoods, key=log_likelihoods.get)
