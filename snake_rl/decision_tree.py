import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import _tree

class Decision_Tree:
    MAX_RULES = 12
    TREE_SCORE_DELTA = 0.03
    WEIGHTS = np.array([5.2, 4.6])  # , 0.1])
    K = np.sum(np.exp(WEIGHTS)) - 2

    def __init__(self, featureset, max_rules=MAX_RULES, tree_score_delta=TREE_SCORE_DELTA, weights=WEIGHTS):
        self.featureset = featureset
        self.max_rules = max_rules
        self.tree_score_delta = tree_score_delta
        self.weights = weights
        self.k = np.sum(np.exp(weights)) - 2

    @classmethod
    def from_XY(cls, X, Y, featureset, max_rules=MAX_RULES, tree_score_delta=TREE_SCORE_DELTA, weights=WEIGHTS):
        self = cls(featureset, max_rules=max_rules, tree_score_delta=tree_score_delta, weights=weights)
        self.tree = None
        self.X = X
        self.Y = Y
        return self

    @classmethod
    def from_tree(cls, tree, featureset, max_rules=MAX_RULES, tree_score_delta=TREE_SCORE_DELTA, weights=WEIGHTS):
        self = cls(featureset, max_rules=max_rules, tree_score_delta=tree_score_delta, weights=weights)
        self.tree = tree
        self.X = None
        self.Y = None
        return self

    """
    Fit a decision tree to the data with certain depth.
    Parameters: X and Y dataframes, and tree depth
    Return: Fitted decision tree 
    """

    def fit_tree(self, depth, X=None, Y=None):

        if (X is not None) and (Y is not None):
            self.X = X
            self.Y = Y

        # create tree
        self.tree = tree.DecisionTreeClassifier(random_state=100, max_depth=depth)
        # fit on data
        self.tree = self.tree.fit(self.X, self.Y)

    """
    Run decision tree on 
    """

    def run_decision_tree(self):

        fin_tree = None

        # for each incrementing number of rules
        for i in range(1, self.max_rules):
            X_train, X_test, y_train, y_test = train_test_split(self.X, self.Y, random_state=0, train_size=.75,
                                                                stratify=self.Y)
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

    """
    Classify the current sample with 
    """

    def classify(self, arr):
        sample = [np.array(self.featureset.expand_arr(arr))]
        r = self.tree.predict_proba(np.array(sample).astype(np.float32))[0, 1]
        return r

    """
    Get with recursion, how bad will the future state be
    """

    def recurse_freq(self, arr):
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

    """
    Returns [d1, d2], when:
        d1 = how bad the current state is
        d2 = how bad the next state will be
    """

    def get_ds(self, sample):
        return np.array(
            [self.recurse_freq(sample), self.recurse_freq(sample[1:])])  # , recurse_freq(tree, sample[2:])])

    """
    Calculate alpha by: 
        alpha = (e^(d1*w1) + e^(d2*w2) - 2) / (e^w1 + e^w2 - 2)
    d1 and d2 = 1, means the state is good, and therefore alpha = 1
    """

    def calc_alpha(self, sample):
        d = self.get_ds(sample)
        a = np.sum(np.exp(d * self.weights)) - 2
        return a / self.k


"""
Beautify Tree
"""


def beauitfy(p):
    return p
    fl = re.findall("\d+\.\d+", p)[0]
    newp = p.split(fl)
    if newp[0][-2] == '=':
        return newp[0][:-2] + " " + str(int(float(fl))) + ")"
    else:
        return newp[0][:-1] + " " + str(int(float(fl))) + ")"


"""
Print tree in a nice way
"""


def get_rules(tree, feature_names, class_names):
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