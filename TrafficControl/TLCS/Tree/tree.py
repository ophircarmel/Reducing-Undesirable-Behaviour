import pickle

import numpy as np
import pandas as pd
import graphviz

from sklearn import tree
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.tree import _tree

from TrafficControl.TLCS.utils_for_tree import FEATURE_DICT, get_list_of_features_from_dict, get_list_of_features_from_dict_snake

from TrafficControl.TLCS.consts import *


def list_to_float(l):
    """
    Turn list of string floats into list of floats
    """
    return [float(k) for k in l]


def get_col_names():
    """
    Get the feature names for the tree.
    """
    # if we are using SNAKE grammar
    if SNAKE:
        example_dict = {
            "W2E" : [0]*8,
            "W2TL" : [0]*8,
            "E2W" : [0]*8,
            "E2TL" : [0]*8,
            "N2S" : [0]*8,
            "N2TL" : [0]*8,
            "S2N" : [0]*8,
            "S2TL" : [0]*8,
            "ACT" : [0]*4
        }
        feats = get_list_of_features_from_dict_snake(example_dict)
        col_names = [feat.to_string() for feat in feats]
        return col_names

    # if we use regular grammar
    col_names = []
    lanes = [
        'W2E', 'W2TL',
        'E2W', 'E2TL',
        'N2S', 'N2TL',
        'S2N', 'S2TL'
    ]
    for lane in lanes:
        col_names += [feat.to_string() for feat in get_list_of_features_from_dict(lane, [], FEATURE_DICT)]

    col_names += ['ACTION = NS', 'ACTION = NSL', 'ACTION = EW', 'ACTION = EWL']
    return col_names

def parse_file(data_file):
    """
    Parse the specifically formatted data file, to which we write while training.
    While training, we write each state-action trace we get into a datafile (des.txt or undes.txt).
    This reads the state-action trace from this datafile, and parses it back into the state-action pair,
     and then extracts the features according to the relevant grammar from this datafile, and puts it in a dataframe.
    param: data_file - the relevant data file.
    """
    df = pd.DataFrame(columns=get_col_names())
    cntr = 0

    with open(data_file, "r") as f:
        while f.readline(): # [
            # this was for checking, NO_FEATS should never be True
            if NO_FEATS:
                entry_dict = \
                    {
                        'W2E': W2E,
                        'W2TL': W2TL,
                        'E2W': E2W,
                        'E2TL': E2TL,
                        'N2S': N2S,
                        'N2TL': N2TL,
                        'S2N': S2N,
                        'S2TL': S2TL,
                    }

                if SNAKE:
                    entry_dict['ACT'] = list_to_float(acts)
                    entry_feats = get_list_of_features_from_dict_snake(entry_dict)

                    entry = []
                    for feat in entry_feats:
                        entry.append(round(feat.evaluate(), 2))
                else:
                    entry_feats = []
                    for key in entry_dict.keys():
                        lane_ID = key
                        lane = list_to_float(entry_dict[key])
                        entry_feats += get_list_of_features_from_dict(lane_ID, lane, FEATURE_DICT)

                    entry = []

                    for feat in entry_feats:
                        entry.append(round(feat.evaluate(), 2))

                    entry += list_to_float(acts)
            else:
                if SNAKE:
                    # if snake grammar, parse and continue
                    entry = f.readline().strip().split(', ')
                    f.readline()
                    f.readline()
                else:
                    # if regular grammar, pase in a meaningful way
                    W2E = f.readline().strip().split(' ,')
                    W2TL = f.readline().strip().split(' ,')
                    E2W = f.readline().strip().split(' ,')
                    E2TL = f.readline().strip().split(' ,')
                    N2S = f.readline().strip().split(' ,')
                    N2TL = f.readline().strip().split(' ,')
                    S2N = f.readline().strip().split(' ,')
                    S2TL = f.readline().strip().split(' ,')
                    acts = f.readline().strip().split(', ')
                    f.readline()  # ]
                    f.readline()  # \n
                    entry =\
                        list_to_float(W2E) + list_to_float(W2TL) +\
                        list_to_float(E2W) + list_to_float(E2TL) +\
                        list_to_float(N2S) + list_to_float(N2TL) +\
                        list_to_float(S2N) + list_to_float(S2TL) +\
                        list_to_float(acts)
            # build the dataframe
            df.loc[len(df)] = entry
            if cntr%1000 == 0:
                print(cntr)
            cntr += 1
    return df

def files_to_labeled_data():
    """
    Get dataframes of the features extracted from the undesirable and desirable files.
    Return X which is the dataframe of the features, for training the tree,
    and Y which is the dataframe of labels.
    """
    # original bad and good
    orig_bad = parse_file(UNDES_FILE)
    orig_good = parse_file(DES_FILE)
    # generate np arrays from files

    # generate dataframe with columns
    X = pd.DataFrame(np.concatenate((orig_bad, orig_good)))
    Y = pd.DataFrame([0] * len(orig_bad) + [1] * len(orig_good))

    # shuffle the dataframes together
    idx = np.random.permutation(X.index)
    X = X.reindex(idx)
    Y = Y.reindex(idx)

    return X, Y


def fit_tree(X, Y, depth):
    """
    Fit a decision tree to the data.
    param: X - the training dataset
    param: Y - the training labels
    param: depth - depth of the tree
    """
    # create tree
    decision_tree = tree.DecisionTreeClassifier(random_state=100, max_depth=depth, class_weight='balanced')
    # fit on data
    decision_tree = decision_tree.fit(X, Y)
    return decision_tree

def run_decision_tree_for_explainability(X, Y):
    """
    Fit trees for different depths, log metrics and the trees themselves.
    param: X - the dataset (before train and test split)
    param: Y - the dataset (after train and test split)
    """
    max_rules = 20
    tree_score_delta = 0.01
    max_score = 0

    trees = {}
    perf = {}

    # for each incrementing number of rules
    for i in range(1, max_rules):
        X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0, train_size=.75,
                                                            stratify=Y)
        # find a tree
        dt = fit_tree(X_train, y_train, i)

        # check if tree gives a near perfect score
        score = dt.score(X_test, y_test)

        y_pred = dt.predict(X_test)

        # get metrics
        attr_dict = {}

        precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred)

        y_test = [item for sublist in y_test.to_numpy().tolist() for item in sublist]
        conf = confusion_matrix(y_test, y_pred.tolist())

        attr_dict["score"] = score
        attr_dict["precision_UNDESIRABLE"] = precision[0]
        attr_dict["precision_DESIRABLE"] = precision[1]
        attr_dict["recall_UNDESIRABLE"] = recall[0]
        attr_dict["recall_DESIRABLE"] = recall[1]
        attr_dict["fscore_UNDESIRABLE"] = fscore[0]
        attr_dict["fscore_DESIRABLE"] = fscore[1]
        attr_dict["support_UNDESIRABLE"] = support[0]
        attr_dict["support_DESIRABLE"] = support[1]

        attr_dict['TRUE_UNDESIRABLE'] = conf[0][0]
        attr_dict['FALSE_DESIRABLE'] = conf[0][1]
        attr_dict['FALSE_UNDESIRABLE'] = conf[1][0]
        attr_dict['TRUE_DESIRABLE'] = conf[1][1]

        perf[i] = attr_dict

        print('precision: {}'.format(precision))
        print('recall: {}'.format(recall))
        print('fscore: {}'.format(fscore))
        print('support: {}'.format(support))


        trees[i] = (dt, score)
        if score > max_score:
            fin_tree = dt
            max_score = score

        if score >= 1.0 - tree_score_delta:
            print('Depth of decision tree is:', i)
            break

        print("Depth is " + str(i) + ", score is " + str(score))

    # log metrics
    pickle.dump(perf, open(PERF_DICT, "wb+"))
    return trees


def get_all_trees(X, Y):
    """
    Create the trees with different depths and save them.
    param: X - the dataset (before train and test split)
    param: Y - the dataset (after train and test split)
    """
    # get the trees
    trees = run_decision_tree_for_explainability(X,Y)

    for depth in trees.keys():
        print('Depth is : ' + str(depth) + ".\nScore is : " + str(trees[depth][1]))

        cur_tree = trees[depth][0]

        # export tree to graphviz
        feature_names = get_col_names()
        r = tree.export_text(cur_tree, feature_names=feature_names)
        print(r)

        # DOT data
        dot_data = tree.export_graphviz(cur_tree, out_file=None,
                                        feature_names=feature_names,
                                        class_names=['Undesirable', 'Desirable'],
                                        filled=True)

        # Draw graph
        graph = graphviz.Source(dot_data, format="png")
        graph.render(TREE_FOLDER + "/depth=" + str(depth))



def run_decision_tree(X, Y):
    """
    Trains the best decision tree.
    param: X - the dataset (before train and test split)
    param: Y - the dataset (after train and test split)
    """
    fin_tree = None
    max_rules = 20
    tree_score_delta = 0.01
    max_score = 0

    # for each incrementing number of rules
    for i in range(1, max_rules):
        X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0, train_size=.75,
                                                            stratify=Y)
        # find a tree
        dt = fit_tree(X_train, y_train, i)

        # check if tree gives a near perfect score
        score = dt.score(X_test, y_test)
        if score > max_score:
            fin_tree = dt
            max_score = score

        if score >= 1.0 - tree_score_delta:
            print('Depth of decision tree is:', i)
            break

        print("Depth is " + str(i) + ", score is " + str(score))

    # export tree
    feature_names = get_col_names()
    r = tree.export_text(fin_tree, feature_names=feature_names)

    # print beautiful tree
    print(r)

    # get tree
    with open(TREE_PTH, "wb+") as f:
        pickle.dump(fin_tree, f)

    return fin_tree


def beauitfy(p):
    """
    Beautify Tree - obsolete function
    """
    return p
    fl = re.findall("\d+\.\d+", p)[0]
    newp = p.split(fl)
    if newp[0][-2] == '=':
        return newp[0][:-2] + " " + str(int(float(fl))) + ")"
    else:
        return newp[0][:-1] + " " + str(int(float(fl))) + ")"


def get_rules(tree, feature_names, class_names):
    """
    Print tree to output log in a pretty way.
    param: tree - the decision tree
    param: feature_names - feature names for the tree.
    param: class_names - names of classes (undesirable and desirable)
    """
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    paths = []
    path = []

    def recurse(node, path, paths):

        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            p1, p2 = list(path), list(path)
            p1 += [f"({name} < {np.ceil(threshold)})"]
            recurse(tree_.children_left[node], p1, paths)
            p2 += [f"({name} > {np.floor(threshold)})"]
            recurse(tree_.children_right[node], p2, paths)
        else:
            path += [(tree_.value[node], tree_.n_node_samples[node])]
            paths += [path]

    recurse(0, path, paths)

    # sort by samples count
    samples_count = [p[-1][1] for p in paths]
    ii = list(np.argsort(samples_count))
    paths = [paths[i] for i in reversed(ii)]

    rules = []
    for path in paths:
        rule = "if "

        for p in path[:-1]:
            if rule != "if ":
                rule += " and "
            rule += str(beauitfy(p))
        rule += " then "
        if class_names is None:
            rule += "response: " + str(np.round(path[-1][0][0][0], 3))
        else:
            classes = path[-1][0][0]
            l = np.argmax(classes)
            rule += f"class: {class_names[l]} (proba: {np.round(100.0 * classes[l] / np.sum(classes), 2)}%)"
        rule += f" | based on {path[-1][1]:,} samples"
        rules += [rule]

    return rules


def pickle_dataframes():
    """
    Load and parse datafiles and save the created dataframes to a pickle file.
    """
    # get the dataframes
    X, Y = files_to_labeled_data()
    # pickle data
    X.to_pickle(X_PKL)
    Y.to_pickle(Y_PKL)


def unpickle_dataframes():
    """
    Get the dataframes from the pickle.
    """
    X = pd.read_pickle(X_PKL)
    Y = pd.read_pickle(Y_PKL)
    print("read pickle")
    return X,Y


if __name__ == '__main__':
    #pickle_dataframes()
    # X, Y = unpickle_dataframes()
    # decision_tree = run_decision_tree(X, Y)
    # filename = 'Properties/PROPERTY 0/_tree.pkl'
    # pickle.dump(decision_tree, open(filename, 'wb'))

    # TREE_FILE_NAME = "Properties/PROPERTY 0/finalized_tree.pkl"
    #
    # fin_tree = pickle.load(open(TREE_FILE_NAME, "rb"))
    # feature_names = get_col_names()
    # r = tree.export_text(fin_tree, feature_names=feature_names)
    # #
    # import graphviz
    #
    # # DOT data
    # dot_data = tree.export_graphviz(fin_tree, out_file=None,
    #                                 feature_names=feature_names,
    #                                 class_names=['Undesirable', 'Desirable'],
    #                                 filled=True)
    #
    # # Draw graph
    # graph = graphviz.Source(dot_data, format="png")
    # graph.render("finalized_tree")

    # after the first run, you can comment this out. This is for creating the pickle files.
    pickle_dataframes()
    X, Y = unpickle_dataframes()
    run_decision_tree(X, Y)
