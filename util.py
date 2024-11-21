from typing import Sequence

def accuracy(true_labels : Sequence[tuple[str, int]], predicted_labels : Sequence[tuple[str, int]]) -> float:
    if len(true_labels) != len(predicted_labels):
        raise ValueError("Length of true_labels and predicted_labels must match")
    if len(true_labels) == 0:
        return 0.0
        
    correct = sum(1 for t, p in zip(true_labels, predicted_labels) if t == p)
    return correct / len(true_labels)

def recall(true_labels : Sequence[tuple[str, int]], predicted_labels : Sequence[tuple[str, int]]) -> float: 
    if len(true_labels) != len(predicted_labels):
        raise ValueError("Length of true_labels and predicted_labels must match")
    if len(true_labels) == 0:
        return 0.0
    
    true_label = true_labels[0][1]

    true_positives = 0
    predicted_positives = 0
    for t, p in zip(true_labels, predicted_labels):
        if p == t and p[1] == true_label:
            true_positives += 1
        if p[1] == true_label:
            predicted_positives += 1
    
    return true_positives / predicted_positives if predicted_positives > 0 else 0.0

def precision(true_labels : Sequence[tuple[str, int]], predicted_labels : Sequence[tuple[str, int]]) -> float: 
    if len(true_labels) != len(predicted_labels):
        raise ValueError("Length of true_labels and predicted_labels must match")
    if len(true_labels) == 0:
        return 0.0
    
    true_label = true_labels[0][1]

    true_positives = 0
    actual_positives = 0
    
    for t, p in zip(true_labels, predicted_labels):
        if p == t and p[1] == true_label:
            true_positives += 1
        if t[1] == true_label:
            actual_positives += 1
    
    return true_positives / actual_positives if actual_positives > 0 else 0.0

def f1(true_labels : Sequence[tuple[str, int]], predicted_labels : Sequence[tuple[str, int]]) -> float:
    if len(true_labels) != len(predicted_labels):
        raise ValueError("Length of true_labels and predicted_labels must match")
    if len(true_labels) == 0:
        return 0.0
    
    true_label = true_labels[0][1]

    true_positives = 0
    actual_positives = 0
    predicted_positives = 0
    
    for t, p in zip(true_labels, predicted_labels):
        if p == t and p[1] == true_label:
            true_positives += 2
        if t[1] == true_label:
            actual_positives += 1
        if p[1] == true_label:
            predicted_positives += 1
    
    return true_positives / (actual_positives + predicted_positives) if actual_positives > 0 else 0.0


