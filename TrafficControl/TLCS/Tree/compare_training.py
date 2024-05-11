import pickle

import matplotlib.pyplot as plt

from TrafficControl.TLCS.consts import *


def change_dict(d):
    """
    Change the metrics dictionaries from {depth : {metric : value}}
    to {metric : values}, where values is the list of the metric values for all depths.
    param d: the dictionary we convert.
    """
    keys = list(d.keys())

    new_d = {}

    # convert dictionary to new format
    for param in d[keys[0]].keys():
        param_lst = []

        for idx in keys:
            param_lst.append(d[idx][param])

        new_d[param] = param_lst
    return new_d



def compare_dicts(dict1, dict2):
    """
    Compare each metrics for two metric dictionaries for trees trained with different grammars.
    param: dict1 - the first dictionary
    param: dict2 - the second dictionary
    """

    # change format
    dict1 = change_dict(dict1)
    dict2 = change_dict(dict2)

    # plot and save each metric comparison
    for param in dict1.keys():
        plt.plot(range(1,len(dict1[param])+1), dict1[param], label = 'Original Grammar')
        plt.plot(range(1,len(dict2[param])+1), dict2[param], label = 'Snake Grammar')
        plt.legend()

        title = 'Comparison of ' + param
        plt.savefig('compare_trees/no_title/' + title + '.png')

        plt.title(title)
        plt.savefig('compare_trees/' + title + '.png')
        plt.show()

        if param == "score":
            print(dict1[param])
            print(dict2[param])




def main():
    # get perfomance dictionaries for both grammars
    PERF1 = PROPERTY + "perf_dict.pkl"
    PERF2 = PROPERTY + "perf_dict_snake.pkl"
    perf1 = pickle.load(open(PERF1, "rb"))
    perf2 = pickle.load(open(PERF2, "rb"))

    # compare performances
    compare_dicts(perf1, perf2)

if __name__ == '__main__':
    main()