import pickle
import sys

import numpy as np
import graphviz
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import _tree

from src.create_trees.create_data_and_labels_from_rules import get_data_and_labels_dataframes_from_file, create_data_and_labels_dataframes_from_file
from src.create_trees.decision_tree import get_rules, Decision_Tree
from src.create_trees.featureset import Featureset
from sklearn import tree


FOLDER = "more_trees/less_than_1_pnt_2/"
PERF_DICT = FOLDER + "perf_dict.pkl"

def fit_tree(X, Y, depth):
    """
    Fit tree to data.
    param: X - data we want to learn on
    param: Y - labels for the data
    param: depth - depth of said tree
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
    param: Y - the labels (before train and test split)
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


    # save metrics
    pickle.dump(perf, open(PERF_DICT, "wb+"))
    return trees


def get_all_trees(X, Y, featureset):
    """
    Fit trees for different depths, on the grammar from 'featureset'
    param: X - the dataset (before train and test split)
    param: Y - the labels (before train and test split)
    param: featureset - the grammar rules
    """
    # create trees
    trees = run_decision_tree_for_explainability(X,Y)

    for depth in trees.keys():
        print('Depth is : ' + str(depth) + ".\nScore is : " + str(trees[depth][1]))

        cur_tree = trees[depth][0]

        # export trees
        feature_names = [feat.name for feat in featureset.feats]
        r = tree.export_text(cur_tree, feature_names=feature_names)
        print(r)

        # DOT data
        dot_data = tree.export_graphviz(cur_tree, out_file=None,
                                        feature_names=feature_names,
                                        class_names=['Undesirable', 'Desirable'],
                                        filled=True)

        # Draw graph
        graph = graphviz.Source(dot_data, format="png")
        graph.render(FOLDER + "depth=" + str(depth) + "_graphivz")


def run_decision_tree(X, Y):
    """
    Get the best decision tree we can get (as close to near perfect score).
    param: X - the dataset (before train and test split)
    param: Y - the labels (before train and test split)
    """
    fin_tree = None
    max_rules = 30
    tree_score_delta = 0.005
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

    # feature_names = get_col_names()
    # r = tree.export_text(fin_tree, feature_names=feature_names)
    #
    # # print beautiful tree
    # print(r)
    # return fin_tree


def beauitfy(p):
    """
    Obsolete
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

def save_col_names():
    """
    Get feature names for the tree visualization.
    """
    featureset_name = sys.argv[3]
    featureset_file = 'featuresets/' + featureset_name + '_featureset.txt'

    featureset = Featureset().parse_file(featureset_file)

    feat_names = [feat.name for feat in featureset.feats]

    # save the feature name
    pickle.dump(feat_names, open(r"featuresets\\" + featureset_name + ".pkl", "wb+"))

if __name__ == '__main__':
    # get ruleset name from the first argument
    ruleset_name = sys.argv[2]
    # get featureset name from the second argument
    featureset_name = sys.argv[3]
    featureset_file = 'featuresets/' + featureset_name + '_featureset.txt'

    # create new files
    x = input("Do you want to create new files?\n")
    if x == 'y':
        # get number of examples in each positive and negative file
        n = input("Input number of undesirable and desirable samples\n")
        size = int(n)
        # get dataframes
        #X, Y = generate_data_and_labels_dataframes(ruleset_name=ruleset_name, featureset_name=featureset_name, size=size)
        X, Y = create_data_and_labels_dataframes_from_file(ruleset_name=ruleset_name, featureset_name=featureset_name, size=size, with_size=False)
        print("Finished generating data")
    else:
        # get dataframes
        X, Y = get_data_and_labels_dataframes_from_file(ruleset_name=ruleset_name)
    # featureset
    featureset = Featureset().parse_file(featureset_file)

    # create tree
    dt = Decision_Tree.from_XY(X=X, Y=Y, featureset=featureset)
    dt.run_decision_tree()
    # get the rules
    rules = get_rules(dt.tree, [feat.name for feat in featureset.feats], ["Positive", "Negative"])
    print(rules[0])

    # save tree
    if input('Save Tree?\n') == 'y':
        pickle.dump(dt.tree, open('trees/' + ruleset_name + '_treefile', 'wb'))
#
# if __name__ == '__main__':
#     # get ruleset name from the first argument
#     ruleset_name = "less_than_1_pnt_2" #sys.argv[2]
#     # get featureset name from the second argument
#     featureset_name = "all_with_res" #sys.argv[3]
#     featureset_file = 'featuresets/' + featureset_name + '_featureset.txt'
#
#     # load the data
#     X, Y = get_data_and_labels_dataframes_from_file(ruleset_name=ruleset_name)
#
#     featureset = Featureset().parse_file(featureset_file)
#
#     # load tree
#     dt = pickle.load(open('trees/less_than_1_pnt_2_treefile', 'rb'))
#
#     #
#     rules = get_rules(dt, [feat.name for feat in featureset.feats], ["Undesirable", "Desirable"])
#     print(rules[0])
#
#     # dot_data = tree.export_graphviz(dt, out_file=None,
#     #                                  feature_names=[feat.name for feat in featureset.feats],
#     #                                  class_names=['Undesirable', 'Desirable'],
#     #                                  filled=True)
#     # #
#     # graph = graphviz.Source(dot_data, format="png")
#     # # graph.render(FOLDER + "depth=" + str(depth) + "_graphivz")
#     # graph.render(FOLDER + "new_tree")
#
#     get_all_trees(X, Y, featureset)
# #
# # if __name__ == '__main__':
# #     # get ruleset name from the first argument
# #     ruleset_name = sys.argv[2]
# #     # get featureset name from the second argument
# #     featureset_name = sys.argv[3]
# #     featureset_file = 'featuresets/' + featureset_name + '_featureset.txt'
# #
# #     # create new files
# #     x = input("Do you want to create new files?\n")
# #     if x == 'y':
# #         # get number of examples in each positive and negative file
# #         n = input("Input number of positive and negative examples\n")
# #         size = int(n)
# #         # get dataframes
# #         #X, Y = generate_data_and_labels_dataframes(ruleset_name=ruleset_name, featureset_name=featureset_name, size=size)
# #         X, Y = create_data_and_labels_dataframes_from_file(ruleset_name=ruleset_name, featureset_name=featureset_name, size=size, with_size=False)
# #         print("Finished generating data")
# #     else:
# #         # get dataframes
# #         X, Y = get_data_and_labels_dataframes_from_file(ruleset_name=ruleset_name)
# #     # featureset
# #     featureset = Featureset().parse_file(featureset_file)
# #
# #     # create tree
# #     dt = Decision_Tree.from_XY(X=X, Y=Y, featureset=featureset)
# #     dt.run_decision_tree()
# #     # get the rules
# #     rules = get_rules(dt.tree, [feat.name for feat in featureset.feats], ["Positive", "Negative"])
# #     print(rules[0])
# #
# #     # save tree
# #     if input('Save Tree?\n') == 'y':
# #         pickle.dump(dt.tree, open('trees/' + ruleset_name + '_treefile', 'wb'))