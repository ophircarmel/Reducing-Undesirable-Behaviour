import sys

import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import _tree

# on cluster or not
if int(sys.argv[1]) == 0:
    import src.create_trees.generate_srs
else:
    import create_trees.generate_srs


class Decision_Tree:
    """
    Decision Tree class.
    attribute: featureset - the grammar rules
    attribute: max_rules - maximum depth
    attribute: tree_score_delta - which minimum scoredelta do we stop at
    attribute: weights - obsolete
    attribute: k - obsolete
    attribute: X - the data we use for the tree
    attribute: Y - the labels for said data
    attribute: tree - the tree itself
    """
    MAX_RULES = 20
    TREE_SCORE_DELTA = 0.005
    WEIGHTS = np.array([5.2, 4.6])  # , 0.1])
    K = np.sum(np.exp(WEIGHTS)) - 2

    def __init__(self, featureset, max_rules=MAX_RULES, tree_score_delta=TREE_SCORE_DELTA, weights=WEIGHTS):
        """
        Initialize attribute
        """
        self.featureset = featureset
        self.max_rules = max_rules
        self.tree_score_delta = tree_score_delta
        self.weights = weights
        self.k = np.sum(np.exp(weights)) - 2

    @classmethod
    def from_XY(cls, X, Y, featureset, max_rules=MAX_RULES, tree_score_delta=TREE_SCORE_DELTA, weights=WEIGHTS):
        """
        Add attributes X,Y along with the regular ones for initialization.
        """
        self = cls(featureset, max_rules=max_rules, tree_score_delta=tree_score_delta, weights=weights)
        self.tree = None
        self.X = X
        self.Y = Y
        return self

    @classmethod
    def from_tree(cls, tree, featureset, max_rules=MAX_RULES, tree_score_delta=TREE_SCORE_DELTA, weights=WEIGHTS):
        """
        Initialize from tree
        """
        self = cls(featureset, max_rules=max_rules, tree_score_delta=tree_score_delta, weights=weights)
        self.tree = tree
        self.X = None
        self.Y = None
        return self




    def fit_tree(self, depth, X=None, Y=None):
        """
            Fit a decision tree to the data with certain depth.
            Parameters: X and Y dataframes, and tree depth
            Return: Fitted decision tree
        """
        
        if (X is not None) and (Y is not None):
            self.X = X
            self.Y = Y
        
        # create tree
        self.tree = tree.DecisionTreeClassifier(random_state=1000, max_depth=depth)
        # fit on data
        self.tree = self.tree.fit(self.X, self.Y)

    def run_decision_tree(self):
        """
        Train decision tree until near perfect score, and then display it.
        """

        fin_tree = None

        # for each incrementing number of rules
        for i in range(1, self.max_rules):
            X_train, X_test, y_train, y_test = train_test_split(self.X, self.Y, random_state=5, train_size=.75, stratify=self.Y)
            # find a tree
            self.fit_tree(i, X_train, y_train)
            dt = self.tree
            fin_tree = dt

            # check if tree gives a near perfect score
            if dt.score(self.X, self.Y) >= 1.0 - self.tree_score_delta:
                print('Depth of decision tree is:', i)
                break

        feature_names = [feat.name for feat in self.featureset.feats]
        r = tree.export_text(fin_tree, feature_names=feature_names)

        # print beautiful tree
        print(r)
        self.tree = fin_tree

    def classify(self, arr, res = (False, 0)):
        """
        Classify current state using the tree.
        """
        sample = [np.array(self.featureset.expand_arr(arr, res))]
        #r = self.tree.predict(np.array(sample).astype(np.float32))[0, 1]
        r = int(self.tree.predict(np.array(sample).astype(np.float32))[0])
        return r

    def recurse_freq(self, arr):
        """
        Obsolete
        """
        if arr[-1] < 10:
            arr[-1] = 10
        if arr[-1] > 24:
            arr[-1] = 24
        if len(arr) < 9:
            raise Exception("Bruh pls no")
        if len(arr) == 10:
            return self.classify(arr)

        last = arr[-1]
        # calculate how bad is the state, by using the probabilities of the next state
        close = (create_trees.generate_srs.DELTA_TOT * create_trees.generate_srs.DELTA_0) \
                * (self.recurse_freq(np.append(arr, [last]))) + \
                (create_trees.generate_srs.DELTA_TOT * create_trees.generate_srs.DELTA_1) / 2 \
                * (self.recurse_freq(np.append(arr, [last + 1])) + self.recurse_freq(np.append(arr, [last - 1]))) + \
                (create_trees.generate_srs.DELTA_TOT * create_trees.generate_srs.DELTA_2) / 2 \
                * (self.recurse_freq(np.append(arr, [last + 2])) + self.recurse_freq(np.append(arr, [last - 2]))) + \
                (create_trees.generate_srs.DELTA_TOT * create_trees.generate_srs.DELTA_3) / 2 \
                * (self.recurse_freq(np.append(arr, [last + 3])) + self.recurse_freq(np.append(arr, [last - 3]))) + \
                (create_trees.generate_srs.DELTA_TOT * create_trees.generate_srs.DELTA_4) / 2 \
                * (self.recurse_freq(np.append(arr, [last + 4])) + self.recurse_freq(np.append(arr, [last - 4])))
        return close / create_trees.generate_srs.DELTA_TOT
        # close = 10 * (
        #     recurse_freq(tree, arr + [last])
        # ) + \
        #         8 * (
        #                 recurse_freq(tree, arr + [last + 1]) +
        #                 recurse_freq(tree, arr + [last + 2]) +
        #                 recurse_freq(tree, arr + [last - 1]) +
        #                 recurse_freq(tree, arr + [last - 2])
        #         ) + \
        #         4 * (
        #                 recurse_freq(tree, arr + [last + 3]) +
        #                 recurse_freq(tree, arr + [last - 3])
        #         ) + \
        #         2 * (
        #                 recurse_freq(tree, arr + [last + 4]) +
        #                 recurse_freq(tree, arr + [last - 4])
        #         ) + \
        #         1 * (
        #                 recurse_freq(tree, arr + [last + 5]) +
        #                 recurse_freq(tree, arr + [last - 5])
        #         )
        return close / 56



    def get_ds(self, sample):
        """
        Obsolete
        """
        return np.array(
            [self.recurse_freq(sample), self.recurse_freq(sample[1:])])  # , recurse_freq(tree, sample[2:])])



    def calc_alpha(self, sample):
        """
        Obsolete
        """
        d = self.get_ds(sample)
        a = np.sum(np.exp(d * self.weights)) - 2
        return a / self.k




def beauitfy(p):
    """
    Obsolete
    """
    return p
    fl = re.findall("\d+\.\d+", p)[0]
    newp = p.split(fl)
    if newp[0][-2] == '=':
        return newp[0][:-2] + " " + str((float(fl))) + ")"
    else:
        return newp[0][:-1] + " " + str((float(fl))) + ")"




def get_rules(tree, feature_names, class_names):
    """
    Print tree in a nice way.
    param: tree - the tree itself
    param: feature_names - grammar
    param: class_names - [Undesirable, Desirable]
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


