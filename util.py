from typing import Sequence

def accuracy(true_labels : set[tuple[str, int]], predicted_labels : Sequence[tuple[str, int]]) -> float:
    
    correct = sum(1 for label in predicted_labels if label in true_labels)
    return correct / len(predicted_labels)

def recall[T](true_labels : Sequence[T], predicted_labels : Sequence[T]) -> float: 
    """ Compute the recall of a model given the true labels and predicted labels
        Note that T is a generic type
    """
    if len(true_labels) != len(predicted_labels):
        raise ValueError("Length of true_labels and predicted_labels must match")
    if len(true_labels) == 0:
        return 0.0

    true_positives = 0
    predicted_positives = 0
    
    for t, p in zip(true_labels, predicted_labels):
        if p == t and t == True:
            true_positives += 1
        if p == True:
            predicted_positives += 1
    
    return true_positives / predicted_positives if predicted_positives > 0 else 0.0

def precision[T](true_labels : Sequence[T], predicted_labels : Sequence[T]) -> float: 
    """ Compute the precision of a model given the true labels and predicted labels
        Note that T is a generic type
    """
    if len(true_labels) != len(predicted_labels):
        raise ValueError("Length of true_labels and predicted_labels must match")
    if len(true_labels) == 0:
        return 0.0

    true_positives = 0
    actual_positives = 0
    
    for t, p in zip(true_labels, predicted_labels):
        if p == t and t == True:
            true_positives += 1
        if t == True:
            actual_positives += 1
    
    return true_positives / actual_positives if actual_positives > 0 else 0.0


