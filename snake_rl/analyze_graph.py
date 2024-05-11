import pickle

import matplotlib.pyplot as plt
import numpy as np

to_cmp = [1.0, 0.75, 0.5, 0.25]

folder = 'save_compared/'

font = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 12,
        }

d = pickle.load(open('compare1.0_0.75_0.5_0.25.pkl', 'rb'))

rewards = d['rewards']
ratios = d['ratios']

x = list(range(len(rewards[1.0])))

for i in to_cmp:
    rm_txt = 'Reward Modifier - ' + str(i)
    if i == 1.0:
        rm_txt += ' (No change)'
    plt.scatter(x, rewards[i], label = rm_txt)
plt.xlabel('Iters')
plt.ylabel('Reward')
plt.ylim(-100, 1100)
title = 'Comparing Final Reward Values between different Reward Modifiers'
# plt.title(title)
plt.legend()
plt.savefig(folder + title.replace(' ', '-') + ".png")
plt.show()

for i in to_cmp:

    mean = np.mean(rewards[i])
    std = np.std(rewards[i])

    rm_txt = 'Reward Modifier = ' + str(i)
    if i == 1.0:
        rm_txt += ' (No change)'

    plt.scatter(x, rewards[i])
    plt.xlabel('Iters')
    plt.ylabel('Reward')
    plt.ylim(-100, 1100)
    plt.text(75, 950, 'Mean=' + str(round(mean, 1)) + '\nStd=' + str(round(std, 1)), fontdict=font)
    title = 'Final Reward Value with ' + rm_txt
    # plt.title(title)
    plt.savefig(folder + title.replace(' ', '-') + ".png")
    plt.show()

for i in to_cmp:
    rm_txt = 'Reward Modifier - ' + str(i)
    if i == 1.0:
        rm_txt += ' (No change)'
    plt.scatter(x, ratios[i], label=rm_txt)
plt.xlabel('Iters')
plt.ylabel('Undesirable to Total Steps Ratio')
plt.ylim(0, 0.4)
title = 'Comparing Final Undesirable Ratio between different Reward Modifiers'
# plt.title(title)
plt.savefig(folder + title.replace(' ', '-') + ".png")
plt.legend()
plt.show()

for i in to_cmp:

    mean = np.mean(ratios[i])
    std = np.std(ratios[i])


    rm_txt = 'Reward Modifier = ' + str(i)
    if i == 1.0:
        rm_txt += ' (No change)'

    plt.scatter(x, ratios[i])
    plt.xlabel('Iters')
    plt.ylabel('Undesirable to Total Steps Ratio')
    plt.ylim(0, 0.4)
    plt.ylim(0, 0.4)
    plt.text(75, 0.35, 'Mean=' + str(round(mean, 4)) + '\nStd=' + str(round(std, 4)), fontdict=font)
    title = 'Final Undesirable Ratio with ' + rm_txt
    # plt.title(title)
    plt.savefig(folder + title.replace(' ', '-') + ".png")
    plt.show()

