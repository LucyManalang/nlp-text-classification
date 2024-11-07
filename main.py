import torch
import argparse
from data import *
from NBBaseline import NBBaseline
from util import *

parser = argparse.ArgumentParser()

parser.add_argument("--data", type=str, help="path to the data folder") 
parser.add_argument("--task", type=str, help="task (imdb or author-id)") 
parser.add_argument("--model", type=str, help="task (baseline, submission, or ...)") 

# Implement these if you find them helpful --- I will train your model's from scratch
parser.add_argument("--save", type=str, help="path to model file to save")
parser.add_argument("--load", type=str, help="path to model file to load")

parser.add_argument("--measure", type=str, help="report the provided measure (acc, precision, recall, f1) over the dev set")
parser.add_argument("--label", action="store_true", help="print out the predicted label of each datapoint in test set, newline separated")

args = parser.parse_args()

if args.task == "imdb":
    dataset = IMDBData(args.data)
    train_data = dataset.get_train_examples()

    data = [] # needed to convert the generator to a list because it was not iterating in the baseline class
    for example in train_data:
        data.append(example)
    
    bayes_baseline = NBBaseline(data)
    print(bayes_baseline.label("elvira mistress of the dark is one of my fav movies , it has every thing you would want in a film , like great one liners , sexy star and a outrageous story ! if you have not seen it , you are missing out on one of the greatest films made . i ca n't wait till her new movie comes out !".split()))
    # TODO: Consult the provided IMDBData class to see how data is stored

elif args.task == "author-id":
    dataset = AuthorIDData(args.data)
    
    # TODO: Consult the provided AuthorIDData class to see how data is stored

if args.measure == "acc":
    #accuracy[T](true_labels : Sequence[T], predicted_labels : Sequence[T]) -> float:
    # unlabeled_data = 
    print("unlabeled data")

    # TODO: Implement this measure

