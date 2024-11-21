import torch
import argparse
from data import *
from NBBaseline import NBBaseline
from TfIdf import TfIdf
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
    dataset = IMDBData("data/imdb/") #having an args.data is redundant
    train_data = list(dataset.get_train_examples())
    if args.model == "baseline":
        model = NBBaseline(train_data)
    elif args.model == "tfidf":
        model = TfIdf(train_data)

    labeled_data = list(dataset.get_dev_examples())
    true_labels = [(" ".join(t[0]), t[1]) for t in labeled_data]
    unlabeled_data = [t[0] for t in labeled_data]
    predicted_labels = [(" ".join(sentence), model.label(sentence)) for sentence in unlabeled_data]
    if args.measure == "acc":
        print("acc: {:.3}".format(100 * accuracy(true_labels, predicted_labels))) 
    elif args.measure == "precision":
        print("precision: {:.3}".format(100 * precision(true_labels, predicted_labels))) 
    elif args.measure == "recall":
        print("recall: {:.3}".format(100 * recall(true_labels, predicted_labels))) 
    elif args.measure == "f1":
        print("f1: {:.3}".format(100 * f1(true_labels, predicted_labels))) 

elif args.task == "author-id":
    dataset = AuthorIDData("/data/author-id/") 
    train_data = list(dataset.get_train_problems()) 
    train_problems = []
    for triplet in train_data:
        problem = []
        for candidate, data in triplet[0].items():
            for sequence in data:
                problem.append((sequence, int(candidate[-5:])))
        train_problems.append(problem)
            
    models = []
    for problem in train_problems:
        if args.model == "baseline":
            models.append(NBBaseline(problem))
        elif args.model == "tfidf":
            models.append(TfIdf(problem))
    
    
    


