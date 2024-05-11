"""
Contains all the functions that creates labeled data for positive and negative example,
when assuming we can create a random starting array.
"""

import os
import pickle
import random
import sys

import numpy as np
import pandas as pd
from sklearn.utils import shuffle

from src.create_trees.featureset import Featureset
from src.create_trees.generate_srs import random_array_generator
from src.create_trees.ruleset import Ruleset

# maximum difference between positive counter and negative counter
MAXDIFF = 1000
# POSTIVE LABEL - when the state is bad
POS_LAB = 0
# NEGATIVE LABEL - when the state is good
NEG_LAB = 1
# samples file
SAMPLE_FILE = "samples.txt"

def shuffle_txt():
    """
    Shuffle sample.txt
    """
    lines = open(SAMPLE_FILE).readlines()
    random.shuffle(lines)
    open(SAMPLE_FILE, 'w').writelines(lines)


def generate_array_from_file():
    """
    Get data from samples.txt.
    A generator function for generating sample states from the file
    """
    lines = open(SAMPLE_FILE).readlines()
    for line in lines:
        line = line.strip()
        arr = [float(part) for part in line.split('  ')]
        yield arr



def file_names(ruleset_name, featureset_name):
    """
    Return file names for rule_file, feature_file, and pos, neg files
    """
    # ruleset and featureset files
    ruleset_file = 'rulesets/' + ruleset_name + '_ruleset.txt'
    featureset_file = 'featuresets/' + featureset_name + '_featureset.txt'
    rules_dir = 'rules_files/' + ruleset_name

    # if the directory does not exist, then create it
    exists = os.path.exists(rules_dir)
    if not exists:
        # Create a new directory because it does not exist
        os.mkdir(rules_dir)

    # get the positive and negative files
    pos_file = rules_dir + '/pos.xlsx'
    neg_file = rules_dir + '/neg.xlsx'

    return ruleset_file, featureset_file, pos_file, neg_file

def create_data_and_labels_dataframes_from_file(ruleset_name, featureset_name, size, with_size=True):
    """
    Create data and labels dataframes from files.
    param: ruleset_name - the ruleset name describing the undesirable behavior
    param: featureset_name - the featureset describing the grammar rules
    param: size - how many entries do we want
    """
    # counters for pos and neg examples
    pos_cntr = 0
    neg_cntr = 0

    # get all the file names
    ruleset_file, featureset_file, pos_file, neg_file = file_names(ruleset_name, featureset_name)

    # create the ruleset and the featureset
    ruleset = Ruleset().parse_file(ruleset_file)
    featureset = Featureset().parse_file(featureset_file)

    # initialize data and labels
    data = []
    labels = []

    for arr in generate_array_from_file():
        # extract the features from the state, including the action
        expanded = featureset.expand_arr(arr)

        # if the undesirable or desirable arrays are more the size, then stop
        if with_size == True:
            if pos_cntr > size and neg_cntr > size:
                break

        # get label of state
        ch = ruleset.check(arr)

        # write to the undesirable or desirable txt file corresponding to the state

        # if pos and neg are about the same size
        if ch and (pos_cntr - neg_cntr < MAXDIFF):
            pos_cntr += 1
            print("CURRENT POS CNTR - ", pos_cntr)
            with open(pos_file[:-5] + '.txt', 'a+') as f:
                f.write(str(expanded))
                f.write("\n")
            data.append(expanded)
            labels.append(POS_LAB)
        # if negative and pos and neg are about the same size
        if (not ch) and (neg_cntr - pos_cntr < MAXDIFF):
            neg_cntr += 1
            with open(neg_file[:-5] + '.txt', 'a+') as f:
                f.write(str(expanded))
            data.append(expanded)
            labels.append(NEG_LAB)

    # data and labels asarray
    data = np.asarray(data)
    labels = np.asarray(labels)

    # shuffle
    p = np.random.permutation(len(data))
    data = data[p]
    labels = labels[p]

    # write to undesirable and desirable (pos and neg) xlsx files
    to_pos_neg_file(data, labels, featureset, pos_file, neg_file)

    # return the dataframes of the features extracted and their labels
    return pd.DataFrame(data), pd.DataFrame(labels)



def generate_data_and_labels_dataframes(ruleset_name, featureset_name, size):
    """
    Obsolete function
    """
    # counters for pos and neg examples
    pos_cntr = 0
    neg_cntr = 0

    ruleset_file, featureset_file, pos_file, neg_file = file_names(ruleset_name, featureset_name)

    # create the ruleset and the featureset
    ruleset = Ruleset().parse_file(ruleset_file)
    featureset = Featureset().parse_file(featureset_file)

    # initialize data and labels
    data = []
    labels = []

    # generate sliding window array
    for arr in random_array_generator():
        # if we work with [10-25] instead of [1 - 2.5]
        arr = [float(a/10) for a in arr]
        # expand the array
        expanded = featureset.expand_arr(arr)

        # if the positive or negative arrays are more the the size, then stop
        if pos_cntr > size and neg_cntr > size:
            break

        # get label of array
        ch = ruleset.check(arr)

        # if positive and pos and neg are about the same size
        if ch and (pos_cntr - neg_cntr < MAXDIFF):
            pos_cntr += 1
            print("CURRENT POS CNTR - ", pos_cntr)
            with open(pos_file[:-5] + '.txt', 'a+') as f:
                f.write(str(expanded))
                f.write("\n")
            data.append(expanded)
            labels.append(POS_LAB)
        # if negative and pos and neg are about the same size
        if (not ch) and (neg_cntr - pos_cntr < MAXDIFF):
            neg_cntr += 1
            with open(neg_file[:-5] + '.txt', 'a+') as f:
                f.write(str(expanded))
            data.append(expanded)
            labels.append(NEG_LAB)

    # data and labels asarray
    data = np.asarray(data)
    labels = np.asarray(labels)

    # shuffle
    p = np.random.permutation(len(data))
    data = data[p]
    labels = labels[p]

    # write to positive and negative files
    to_pos_neg_file(data, labels, featureset, pos_file, neg_file)

    return pd.DataFrame(data), pd.DataFrame(labels)


def to_pos_neg_file(data, labels, featureset, pos_filename, neg_filename):
    """
    Write the state-action pairs with the features extracted into either
    a desirable (positive) file or an undesirable (negative) file
    param: data - the data to put in the files
    param: labels - labels of said data
    param: featureset - contains the features defined in the grammar
    param: pos_filename - name of DESIRABLE behavior file
    param: neg_filename - name of UNDESIRABLE behavior file
    """
    # get only desirables
    pos = np.around(data[labels == POS_LAB], decimals=1)

    # create excels
    df = pd.DataFrame(pos, index=[str(integ) for integ in range(len(pos))],
                      columns=[feat.name for feat in featureset.feats])
    df.to_excel(pos_filename, index=True, header=True)

    # get only negatives
    neg = np.around(data[labels == NEG_LAB], decimals=1)

    # create excel
    df = pd.DataFrame(neg, index=[str(integ) for integ in range(len(neg))],
                      columns=[feat.name for feat in featureset.feats])
    df.to_excel(neg_filename, index=True, header=True)


def get_data_and_labels_dataframes_from_file(ruleset_name):
    """
    Get data and label dataframes from files
    param: ruleset_name - describes the undesirable behavior
    """
    # get positive and negative file names
    pos_file = 'rules_files/' + ruleset_name + '/pos.xlsx'
    neg_file = 'rules_files/' + ruleset_name + '/neg.xlsx'
    pkl_X_file = 'rules_files/' + ruleset_name + '/X.pkl'
    pkl_Y_file = 'rules_files/' + ruleset_name + '/Y.pkl'

    if os.path.isfile(pkl_X_file) and os.path.isfile(pkl_Y_file):
        with open(pkl_X_file, "rb") as f:
            X = pickle.load(f)
        with open(pkl_Y_file, "rb") as f:
            Y = pickle.load(f)

        return X, Y


    # read from excel files
    X1 = pd.read_excel(pos_file)
    X2 = pd.read_excel(neg_file)

    # combine the positive and negative examples
    X = pd.concat([X1, X2])

    # create labels
    Y = pd.DataFrame([POS_LAB]*len(X1) + [NEG_LAB]*len(X2))

    # shuffle together
    X, Y = shuffle(X, Y, random_state=0)
    # drop temp columns
    X.drop(columns=X.columns[0], axis=1, inplace=True)
    #Y.drop(columns=Y.columns[0], axis=1, inplace=True)

    with open(pkl_X_file, "wb") as f:
        pickle.dump(X, f)
    with open(pkl_Y_file, "wb") as f:
        pickle.dump(Y, f)

    return X, Y