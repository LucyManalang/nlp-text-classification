from typing import Sequence

def accuracy[T](true_labels : Sequence[T], predicted_labels : Sequence[T]) -> float: 
    """ Compute the accuracy of a model given the true labels and predicted labels
        Note that T is a generic type
    """
    true = 0
    false = 0
    

    #tp + tn / tp + tn + fp + fn = t / t + f
    return true / (true + false)

def f1score[T](true_labels : Sequence[T], predicted_labels : Sequence[T]) -> float: 
    """ Compute the f1 of a model given the true labels and predicted labels
        Note that T is a generic type
    """
    return 0.0

def precision[T](true_labels : Sequence[T], predicted_labels : Sequence[T]) -> float: 
    """ Compute the precision of a model given the true labels and predicted labels
        Note that T is a generic type
    """
    return 0.0

def recall[T](true_labels : Sequence[T], predicted_labels : Sequence[T]) -> float: 
    """ Compute the recall of a model given the true labels and predicted labels
        Note that T is a generic type
    """
    return 0.0


# TODO: Be sure to implement the above measures, as well as a way to construct a confusion matrix 
# for your report
