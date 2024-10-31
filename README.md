# HW 2: Text Classification

## Part 0: Familiarize yourself with the tasks

### Sentiment Analysis (IMDB)
For this task, you will be extending your experience with Sentiment Analysis from your in-class activity, this time with the full training split of the IMDB corpus from [Maas et al.](https://ai.stanford.edu/~amaas/data/sentiment/). 

The task revolves around assigning the label *pos* to positive reviews and the label *neg* to negative reviews. Reviews are represented as text, so you'll need to do some NLP!

You are given ~20,000 examples (balanced between positive and negative) as training data with ~5000 examples as a development set. You are provided training examples with no labels which I will ask you to label.

### Author-ID
For this task, you will be performing author identification using data from the shared task from [PAN 2018](https://pan.webis.de/clef18/pan18-web/authorship-attribution.html). Given a handful of documents from a variety of candidate authors, you will have to identify the most likely author for each of a number of target documents.

This task is distinct from the sentiment analysis task in the sense that you will have to train a new model over the cadidate author documents for each task! Make sure you're comfortable with the structure of the data before training models on this task.

### Provided Code & Intended inputs/outputs

Be sure to review the provided code, as well as the Naive Bayes activity, before writing any new code!

`main.py` should store code to manage command-line arguments and produce the correct outputs. Be sure to read the help strings for each of the provided command-line arguments to make sure the intended arguments are recognized. `--save` and `--load` are provided as command-line arguments, but no implementation is given to facilitate you saving and loading any kind of model structure you construct in part 2. Implementing these methods is not required, but doing so should be fairly straightforward and will make doing analyses on a particular model relatively straightforward.

`data.py` is provided to give you code to load the datasets. You can write a fully functional solution using the classes as provided, but you are free to modify this file as you see fit. Make sure you understand how the file structure and data is meant to be organized. 

`util.py` has templates for writing evaluation functions. You may modify the provided signatures as you see fit.

`NBBaseline.py` provides a template for writing the Naive Bayes classifier baseline in Part 1. Complete it, and modify as you see fit.

The `data/` folder stores the provided data. You may peruse it to understand the format of the data. 

The goal is to complete your code such that a a call like

`python main.py --data data/imdb --task imdb --measure acc --model baseline`

Will print out something of the form

`acc: 12.345`

i.e., the name of the measure, a colon and space, and the accuracy with 3 digits past the decimal point. Note that this is not a real value you should expect to get.

and 

`python main.py --data data/imdb --task imdb --model baseline --label`

Will print out predicted labels for each item in the test set, newline separated.

Labels should be either *pos* or *neg* for sentiment analysis and something of the form *candidateXXXXX* for the author-id task.

The output can (and will, during testing) be redirected into a file to store your model's predictions over the data!

## Part 1: Baselines and Evaluations

1. Implement and train Naive Bayes Classifiers (with Laplace smoothing, as discussed in class) for each task

2. Compute accuracy over the training and development sets. You should get ~84% accuracy on the IMDB development set using the baseline.

3. Construct confusion matrices for your classifiers 

4. Compute Precision, Recall, and F1 over the development set. 

**Part 1 will be due in advance of Parts 2 & 3**. Check the course website for the offical due dates for this semester.

## Part 2: Feature Engineering and Model Selection

Using the techniques we've learned in class (Logistic Regression, n-gram modeling, word embedding techniques, etc.) develop a model distinct from the baseline. Ideally, it performs at or above basline performance on at least one of the tasks. 

Additional points may be awarded for the model submission that performs best on the development set or a withheld test set. 

## Part 3: Reporting

Similar to HW1, you'll be asked to write a report to record and discuss your experiments. I will be looking for:

1. A discussion of the baseline implementation as well as your model. Include details about the model architecture, any additional tools you used, the motivation behind it's design, and how it was trained, if applicable. It should be in enough detail that I could, without consulting you directly, re-implement your model simply by reading your report. While brief pseudocode is acceptable, present things in natural language when possible. 

2. A reporting of the results of your analyses. Include a table summarizing the accuracies and F1 scores (and precision and recalls) for the baseline as well as your model. For the IMDB task, present an analysis over the full dataset. For the author ID task, present summary results across all of the problems in this section.

3. An analysis of the results of your model. For your discussion of the IMDB results, include a discussion of the confusion matrix of your model and interpret the results: Is performance comparable across labels? Look through some of the cases where the model succeeded and failed and see if you can pick out any patterns. For your discussion of the Author ID task, you may present results from individual problems if you find them illustrative of your model's performance. 

Submit your report as a typeset pdf file named *hw2-report.pdf* in the top-level directory of your submission's repository to make grading easy. 

Be sure to incorporate feedback provided on your HW1 reports when writing your second report. Be sure to include an attempt at all 3 required parts of the report, and don't let writing the report be an afterthought! 
