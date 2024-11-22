from typing import Sequence, Iterable

def accuracy(true_labels : Sequence[tuple[Iterable[str], int]], predicted_labels : Sequence[tuple[Iterable[str], int]]) -> float:
    if len(true_labels) != len(predicted_labels):
        raise ValueError("Length of true_labels and predicted_labels must match")
    if len(true_labels) == 0:
        raise ValueError("Length of true_labels must be greater than 0")
    
    correct = sum(1 for t, p in zip(true_labels, predicted_labels) if t == p)

    return correct / len(true_labels)

def recall(true_labels : Sequence[tuple[Iterable[str], int]], predicted_labels : Sequence[tuple[Iterable[str], int]]) -> float:
    if len(true_labels) != len(predicted_labels):
        raise ValueError("Length of true_labels and predicted_labels must match")
    if len(true_labels) == 0:
        raise ValueError("Length of true_labels must be greater than 0")
    
    true_label = true_labels[0][1]

    true_positives = 0
    predicted_positives = 0
    for t, p in zip(true_labels, predicted_labels):
        if p == t and p[1] == true_label:
            true_positives += 1
        if p[1] == true_label:
            predicted_positives += 1
    
    return true_positives / predicted_positives if predicted_positives > 0 else 0.0

def precision(true_labels : Sequence[tuple[Iterable[str], int]], predicted_labels : Sequence[tuple[Iterable[str], int]]) -> float:
    if len(true_labels) != len(predicted_labels):
        raise ValueError("Length of true_labels and predicted_labels must match")
    if len(true_labels) == 0:
        raise ValueError("Length of true_labels must be greater than 0")
    
    true_label = true_labels[0][1]

    true_positives = 0
    actual_positives = 0
    
    for t, p in zip(true_labels, predicted_labels):
        if p == t and p[1] == true_label:
            true_positives += 1
        if t[1] == true_label:
            actual_positives += 1
    
    return true_positives / actual_positives if actual_positives > 0 else 0.0

def f1(true_labels : Sequence[tuple[Iterable[str], int]], predicted_labels : Sequence[tuple[Iterable[str], int]]) -> float:
    if len(true_labels) != len(predicted_labels):
        raise ValueError("Length of true_labels and predicted_labels must match")
    if len(true_labels) == 0:
        raise ValueError("Length of true_labels must be greater than 0")
    
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
    
    return true_positives / (actual_positives + predicted_positives) 

def debug(true_labels : Sequence[tuple[Iterable[str], int]], predicted_labels : Sequence[tuple[Iterable[str], int]]) -> float:
    if len(true_labels) != len(predicted_labels):
        raise ValueError("Length of true_labels and predicted_labels must match")
    if len(true_labels) == 0:
        raise ValueError("Length of true_labels must be greater than 0")
    
    true_label = true_labels[0][1]

    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0
    
    for t, p in zip(true_labels, predicted_labels):
        if p == t and p[1] == true_label:
            true_positives += 1
        elif p != t and p[1] == true_label:
            false_positives += 1
        elif p != t and p[1] != true_label:
            false_negatives += 1
        else:
            true_negatives += 1
    
    return (f"true positives: {true_positives}, true negatives: {true_negatives}, false positives: {false_positives}, false negatives: {false_negatives}")



