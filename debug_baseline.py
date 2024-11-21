from NBBaseline import NBBaseline
from data import *
dataset = IMDBData("data/imdb/")
train_data = list(dataset.get_train_examples())
model = NBBaseline(train_data)
%history -f debug_baseline.py
