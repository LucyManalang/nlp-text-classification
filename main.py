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
    train_data = list(dataset.get_train_problems()) # this type is rediculous
    train_data = [item for sublist in train_data for item in sublist]


if args.model == "baseline":
    model = NBBaseline(train_data)
elif args.model == "tfidf":
    model = TfIdf(train_data)

if args.measure == "acc":
    labeled_data = list(dataset.get_dev_examples())
    true_labels = set([(" ".join(t[0]), t[1]) for t in labeled_data])

    unlabeled_data = [t[0] for t in labeled_data]
    predicted_labels = [(" ".join(sentence), model.label(sentence)) for sentence in unlabeled_data]
    print(model.label("enchanted april is a tone poem , an impressionist painting , a masterpiece of conveying a message with few words . it has been one of my 10 favorite films since it came out . i continue to wait , albeit less patiently , for the film to come out in dvd format . apparently , i am not alone . if parent company amazon 's listings are correct , there are many people who want this title in dvd format . many people want to go to italy with this cast and this script . many people want to keep a permanent copy of this film in their libraries . the cast is spectacular , the cinematography and direction impeccable . the film is a definite keeper . many have already asked . please add our names to the list ."))
    print("acc: {:.3}".format(100 * accuracy(true_labels, predicted_labels)))
