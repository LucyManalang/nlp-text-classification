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
    elif args.model == "tfidf":
        model = TfIdf(train_data, scikit_learn=True)

    labeled_data = list(dataset.get_dev_examples())
    true_labels = [(t[0], t[1]) for t in labeled_data]
    predicted_labels = [(t[0], model.label(t[0])) for t in labeled_data]
    if args.measure == "acc":
        print("acc: {:.3}".format(100 * accuracy(true_labels, predicted_labels))) 
    elif args.measure == "precision":
        print("precision: {:.3}".format(100 * precision(true_labels, predicted_labels))) 
    elif args.measure == "recall":
        print("recall: {:.3}".format(100 * recall(true_labels, predicted_labels))) 
    elif args.measure == "f1":
        print("f1: {:.3}".format(100 * f1(true_labels, predicted_labels))) 
    elif args.measure == "debug":
        print(debug(true_labels, predicted_labels))

elif args.task == "author-id":
    dataset = AuthorIDData("data/author-id/") 
    train_data = list(dataset.get_train_problems()) 

    # stores training data in a list by problem
    train_problems = []
    for triplet in train_data:
        problem = []
        for candidate, data in triplet[0].items():
            for sequence in data:
                problem.append((sequence, int(candidate[-5:]) - 1))
        train_problems.append(problem)
            
    # stores models in a list by problem, the index of the model corresponds to the index of the problem
    models = []
    for problem in train_problems:
        if args.model == "baseline":
            models.append(NBBaseline(problem))
        elif args.model == "tfidf":
            models.append(TfIdf(problem))
        elif args.model == "tfidf-scikit":
            models.append(TfIdf(problem, scikit_learn=True))
    
    # stores test data in a list by problem, the index of the problem corresponds to the index of the model and problem
    test_data = list(dataset.get_test_problems())
    test_problems = []
    for triplet in test_data:
        problem = []
        for candidate, data in triplet[0].items():
            for sequence in data:
                problem.append((sequence, int(candidate[-5:]) - 1))
        test_problems.append(problem)
    
    total_avg = 0
    
    # the index-based storage of the test data and the models allows us to zip them together
    for problem, model in zip(test_problems, models):
        true_labels = [(t[0], t[1]) for t in problem]
        true_candidates = {t[1] : [] for t in true_labels}
        for t in true_labels:
            true_candidates[t[1]].append(t)
        
        problem_avg = 0

        # measure the performance of the model (for each problem)
        for candidate in true_candidates.values():
            predicted_labels = [(t[0], model.label(t[0])) for t in candidate]
            if args.measure == "acc":
                print("acc: {:.3}".format(100 * accuracy(candidate, predicted_labels)))
                # problem_avg += accuracy(candidate, predicted_labels)
            if args.measure == "precision":
                print("precision: {:.3}".format(100 * precision(candidate, predicted_labels)))
                # problem_avg += precision(candidate, predicted_labels)
            if args.measure == "recall":
                print("recall: {:.3}".format(100 * recall(candidate, predicted_labels)))
                # problem_avg += recall(candidate, predicted_labels)
            if args.measure == "f1":
                problem_avg += f1(candidate, predicted_labels)
            if args.measure == "debug":
                print(debug(candidate, predicted_labels))
        total_avg += problem_avg / len(true_candidates)
    total_avg /= len(test_problems)

    # f1 is macro-averaged across all problems, so it is calculated sepera
    if args.measure == "f1":
        print("f1: {:.3}".format(100 * total_avg))
    
    #used for macro-averaging across problems
    # print(args.measure, ": ", total_avg)
                

