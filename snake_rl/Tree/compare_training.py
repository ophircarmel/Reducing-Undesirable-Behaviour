import pickle

import matplotlib.pyplot as plt


def compare_dicts(dict1, dict2):
    params = list(dict1[1].keys())

    for param in params:
        steps1 = sorted(list(dict1.keys()))
        steps2 = sorted(list(dict2.keys()))

        l1 = [dict1[step][param] for step in steps1]
        l2 = [dict2[step][param] for step in steps2]

        plt.plot(steps1, l1, label='Original Grammar')
        plt.plot(steps2, l2, label='Traffic Grammar')
        plt.plot(list(range(1, 20)), [0.99]*19, label="0.99")

        plt.legend()
        plt.xlabel("Depth Of Tree")
        plt.ylabel("Value of Parameter")

        title = 'Comparison of ' + param

        plt.title(title)

        #plt.savefig('compare_trees/' + title)

        plt.show()

if __name__ == '__main__':
    d1 = pickle.load(open("Properties/PROPERTY 1/dict.pkl", 'rb'))
    d2 = pickle.load(open("Properties/PROPERTY 1/dict_traffic.pkl", 'rb'))

    compare_dicts(d1, d2)