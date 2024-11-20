from NBBaseline import NBBaseline
from data import *
dataset = IMDBData("data/imdb/")
train_data = list(dataset.get_train_examples())
model = NBBaseline(train_data)
labeled_data = list(dataset.get_dev_examples())
true_labels = set([(" ".join(t[0]), t[1]) for t in labeled_data])
true_data = set([" ".join(t[0]) for t in labeled_data)
true_data = set([" ".join(t[0]) for t in labeled_data])
unlabeled_data = [sentence.split() for sentence in list(dataset.get_test_examples())]
relevant_data = [sentence for sentence in unlabeled_data if " ".join(sentence) in true_data] # because dev data does not contain all of test data, filtering out relevant data is necessary
predicted_labels = [(" ".join(sentence), model.label(sentence)) for sentence in relevant_data]
print("acc: {:.3}".format(100 *accuracy(true_labels, predicted_labels)))
from util import *
print("acc: {:.3}".format(100 *accuracy(true_labels, predicted_labels)))
%history -f script.py
