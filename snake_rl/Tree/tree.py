import pickle

import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
import graphviz
from sklearn.tree import _tree

from Tree.utils_for_tree import FEATURE_DICT, get_list_of_features_from_dict, get_list_of_features_from_dict_traffic

PROPERTY = "Properties/PROPERTY 1/"
X_PKL = PROPERTY + "X_new.pkl"
Y_PKL = PROPERTY + "Y_new.pkl"
UNDES_FILE = PROPERTY + 'undes.txt'
DES_FILE = PROPERTY + 'des.txt'
TREE_PTH = PROPERTY + 'new_tree.pkl'
DICT_PTH = PROPERTY + 'dict.pkl'

# if we want to run with the Traffic grammar - change to 'True'
TRAFFIC = False
if TRAFFIC:
    X_PKL = PROPERTY + "X_traffic.pkl"
    Y_PKL = PROPERTY + "Y_traffic.pkl"
    UNDES_FILE = PROPERTY + 'undes_traffic.txt'
    DES_FILE = PROPERTY + 'des_traffic.txt'
    TREE_PTH = PROPERTY + 'tree_traffic.pkl'
    DICT_PTH = PROPERTY + 'dict_traffic.pkl'



def list_to_float(l):
    """
    Turn list of string floats into list of floats
    """
    return [float(k) for k in l]



def get_col_names():
    """
    Return the list of feature names for snake grammar.
    """
    feats = get_list_of_features_from_dict([])
    names = [feat.name() for feat in feats]
    return names

def get_col_names_traffic():
    """
    Return the list of feature names for Traffic grammar.
    """
    example_dict = {
        'APL' : [0, 0, 0, 0],
        'OBS' : [0, 0, 0, 0],
        'DIR' : [0, 0, 0, 0],
        'ACT' : [0, 0, 0, 0]
    }
    feats = get_list_of_features_from_dict_traffic(example_dict)
    names = [feat.name() for feat in feats]
    return names



def parse_file(data_file):
    """
    Parse data file to dataframe
    param: data_file - file to turn into dataframe.
    """
    if TRAFFIC:
        col_names = get_col_names_traffic()
    else:
        col_names = get_col_names()
    df = pd.DataFrame(columns=col_names)
    cntr = 0

    # build dataframe line by line
    with open(data_file, "r") as f:
        line = f.readline()
        while line:
            entry_strings = (line[1:-2]).split(', ')
            entry = [float(ent) for ent in entry_strings]
            df.loc[len(df)] = entry
            if cntr%1000 == 0:
                print(cntr)
            cntr += 1
            line = f.readline()

    return df

def files_to_labeled_data():
    """
    Get labeled dataframes from data files
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

"""
Fit tree
"""
def fit_tree(X, Y, depth):
    # create tree
    decision_tree = tree.DecisionTreeClassifier(random_state=1000, max_depth=depth, class_weight='balanced')
    # fit on data
    decision_tree = decision_tree.fit(X, Y)
    return decision_tree

"""
"""
def run_decision_tree_for_explainability(X, Y):
    """
    Train trees for each depth.
    Return a dictionary of {depth : tree}
    param: X - features
    param: Y - labels
    """
    max_rules = 20
    tree_score_delta = 0.0
    max_score = 0

    trees = {}
    d = {}

    # for each incrementing number of rules
    for i in range(1, max_rules):
        # split into train and test
        X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0, train_size=.75,
                                                            stratify=Y)
        # find a tree
        dt = fit_tree(X_train, y_train, i)

        # get score
        score = dt.score(X_test, y_test)

        y_pred = dt.predict(X_test)

        # get metrics
        precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred)


        y_test = [item for sublist in y_test.to_numpy().tolist() for item in sublist]

        print('precision: {}'.format(precision))
        print('recall: {}'.format(recall))
        print('fscore: {}'.format(fscore))
        print('support: {}'.format(support))

        conf_mat = confusion_matrix(y_test, y_pred.tolist())
        tn, fp, fn, tp = conf_mat.ravel()

        # attribute dictionary
        attrib = {
            'score' : score,
            'precision_UNDES' : precision[0],
            'precision_DES' : precision[1],
            'recall_UNDES' : recall[0],
            'recall_DES' : recall[1],
            'fscore_UNDES': fscore[0],
            'fscore_DES': fscore[1],
            'support_UNDES': support[0],
            'support_DES': support[1],
            'TRUE_UNDES': tn,
            'TRUE_DES': tp,
            'FALSE_UNDES' : fn,
            'FALSE_DES' : fp
        }

        # put current attributes into the depth
        d[i] = attrib

        # put depth -> (tree, score) in dictionary
        trees[i] = (dt, score)
        if score > max_score:
            fin_tree = dt
            max_score = score

        if score >= 1.0 - tree_score_delta:
            print('Depth of decision tree is:', i)
            break

        print("Depth is " + str(i) + ", score is " + str(score))

    # save attributes
    pickle.dump(d, open(DICT_PTH, 'wb+'))
    return trees


def get_all_trees(X, Y):
    """
    Run trees for all depths
    param: X - the features
    param: Y - labels
    """
    # get all the trees into a dictionary of {depth: tree}
    trees = run_decision_tree_for_explainability(X,Y)

    print("Printing Trees\n")
    for depth in trees.keys():
        # print each depth
        print('Depth is : ' + str(depth) + ", Score is : " + str(trees[depth][1]))

        cur_tree = trees[depth][0]

        if TRAFFIC:
            colnames = get_col_names_traffic
        else:
            colnames = get_col_names

        feature_names = colnames()
        #r = tree.export_text(cur_tree, feature_names=feature_names)
        #print(r)

        # rules = get_rules(cur_tree, feature_names, ["Undesirable", "Desirable"])
        # print(rules[0])

        # DOT data
        dot_data = tree.export_graphviz(cur_tree, out_file=None,
                                         feature_names=feature_names,
                                         class_names=['Undesirable', 'Desirable'],
                                         filled=True)
        #
        # Draw tree
        graph = graphviz.Source(dot_data, format="png")

        if TRAFFIC:
            # using Traffic-Control grammar
            graph.render("trees/traffic_trees_new/depth=" + str(depth))
        else:
            graph.render("trees/trees_new/depth=" + str(depth))


"""
Runs the decision tree
"""
def run_decision_tree(X, Y):
    fin_tree = None
    max_rules = 20
    tree_score_delta = 0.001
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
            print("Depth is " + str(i) + ", score is " + str(score))
            break

        print("Depth is " + str(i) + ", score is " + str(score))

    if TRAFFIC:
        colnames = get_col_names_traffic
    else:
        colnames = get_col_names

    feature_names = colnames()
    r = tree.export_text(fin_tree, feature_names=feature_names)

    # print beautiful tree
    print(r)
    return fin_tree


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
def get_rules(dt, feature_names, class_names):
    tree_ = dt.tree_
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
    X, Y = files_to_labeled_data()
    X.to_pickle(X_PKL)
    Y.to_pickle(Y_PKL)


def unpickle_dataframes():
    X = pd.read_pickle(X_PKL)
    Y = pd.read_pickle(Y_PKL)
    print("read pickle")
    return X,Y


"""
Running flow
"""
def running_flow(to_pickle):
    if to_pickle:
        pickle_dataframes()
    X, Y = unpickle_dataframes()
    decision_tree = run_decision_tree(X, Y)
    filename = TREE_PTH
    pickle.dump(decision_tree, open(filename, 'wb'))

    fin_tree = pickle.load(open(TREE_PTH, "rb"))
    feature_names = get_col_names()
    #r = tree.export_text(fin_tree, feature_names=feature_names)
    #print(r)

    # DOT data
    dot_data = tree.export_graphviz(fin_tree, out_file=None,
                                    feature_names=feature_names,
                                    class_names=['Undesirable', 'Desirable'],
                                    filled=True)

    # Draw graph
    graph = graphviz.Source(dot_data, format="png")
    graph.render("decision_tree_graphivz")


def running_flow_all_trees(to_pickle):
    """
    Train trees.
    param: to_pickle - if we want to pickle the dataframes.
    """
    if to_pickle:
        # we must pickle before unpickling
        pickle_dataframes()
    # unpickle dataframes - if this is error, run with to_pickle=True
    X, Y = unpickle_dataframes()
    # get all the trees
    get_all_trees(X, Y)


def get_aurora_tree():
    """
    Obsolete
    """
    # you may need to change this path in order to save correctly
    tree_pth = r"PCC-RL-GRAPHING\PCC-RL-master\src\create_trees\trees\less_than_1_pnt_2_treefile"
    # you may need to change this path in order to save correctly
    featnames_pth = r"PCC-RL-GRAPHING\PCC-RL-master\src\create_trees\featuresets\all_with_res.pkl"


    fin_tree = pickle.load(open(tree_pth, "rb"))
    feature_names = pickle.load(open(featnames_pth, "rb"))
    # r = tree.export_text(fin_tree, feature_names=feature_names)
    # print(r)

    # DOT data
    dot_data = tree.export_graphviz(fin_tree, out_file=None,
                                    feature_names=feature_names,
                                    class_names=['Undesirable', 'Desirable'],
                                    filled=True)

    graph = graphviz.Source(dot_data, format="png")
    graph.render("aurora_tree")

if __name__ == '__main__':
    #X, Y = unpickle_dataframes()
    running_flow_all_trees(to_pickle=True)
    #running_flow(to_pickle=False)
    #get_aurora_tree()
