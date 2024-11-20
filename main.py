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
    dataset = IMDBData(args.data)
    train_data = list(dataset.get_train_examples())
elif args.task == "author-id":
    dataset = AuthorIDData(args.data)
    # TODO: Consult the provided AuthorIDData class to see how data is stored


if args.model == "baseline":
    model = NBBaseline(train_data)
elif args.model == "tfidf":
    model = TfIdf(train_data)

if args.measure == "acc":
    labeled_data = list(dataset.get_dev_examples())
    true_labels = set([(" ".join(t[0]), t[1]) for t in labeled_data])

    unlabeled_data = [t[0] for t in labeled_data]
    # predicted_labels = [(" ".join(sentence), model.label(sentence)) for sentence in unlabeled_data]
    # model.label("this is a postive sentence good amazing loved")
    # print("acc: {:.3}".format(100 * accuracy(true_labels, predicted_labels)))
