import datetime
import json
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

tree = np
MODIFIER = True
UNDESIRABLE_BEHAVIOR = True
def undesirability(a):
    pass


def reward_modifier(state, action):
    outcome = tree.classify((state, action))

    if outcome == UNDESIRABLE_BEHAVIOR:
        return MODIFIER
    else:
        return 1


def reward_modifier(state, action):
    outcome = tree.classify((state, action))

    if outcome == UNDESIRABLE_BEHAVIOR:
        return MODIFIER * undesirability((state, action))
    else:
        return 1


def modify_reward(reward, state, action):
    modifier = reward_modifier(state, action)

    if reward >= 0:
        reward = reward * modifier
    else:
        reward = reward * (1/modifier)

    return reward


def compare_models_from_file(file):
    with open(file, 'r') as f:
        dict_to_list = {}
        for line in f:
            if not line.startswith('{'):
                continue
            d = json.loads(line)
            for key in d.keys():
                if key in dict_to_list.keys():
                    dict_to_list[key].append(d[key])
                else:
                    dict_to_list[key] = [d[key]]

    for key in dict_to_list.keys():
        plt.hist(dict_to_list[key], bins=50, alpha=0.5, label=key)
        print(key, ": Mean = ", np.mean(dict_to_list[key]), " Var = ", np.var(dict_to_list[key]))
    plt.gca().set(title='Ratio Histogram', ylabel='Ratio')
    plt.legend(loc='upper right')
    plt.show()

    for key in dict_to_list.keys():
        plt.scatter([i for i in range(len(dict_to_list[key]))], dict_to_list[key], label=key)
    plt.gca().set(title='Ratio Histogram', ylabel='Ratio')
    plt.legend(loc='upper right')
    plt.show()

    print(dict_to_list)


def compare_models_from_file_comparative(models, to_cmp):
    folder = 'TLCS/save_compared/no_title/'

    font = {'family': 'serif',
            'color': 'darkred',
            'weight': 'normal',
            'size': 12,
            }

    d = pickle.load(open('TLCS/compare_models_55-68-69-70.pkl', 'rb'))

    rewards = d['rewards']
    ratios = d['ratios']

    x = list(range(len(rewards[models[0]])))

    for i in models:
        rm_txt = 'Reward Modifier - ' + str(to_cmp[i])
        if to_cmp[i] == 1.0:
            rm_txt += ' (No change)'
        plt.scatter(x, rewards[i], label=rm_txt)
    plt.xlabel('Iters')
    plt.ylabel('Reward')
    plt.ylim(-5500, -3000)
    title = 'Comparing Final Reward Values between different Reward Modifiers'
    # plt.title(title)
    plt.legend()
    plt.savefig(folder + title.replace(' ', '-') + ".png")
    plt.show()

    for i in models:

        mean = np.mean(rewards[i])
        std = np.std(rewards[i])

        rm_txt = 'Reward Modifier = ' + str(to_cmp[i])
        if to_cmp[i] == 1.0:
            rm_txt += ' (No change)'

        plt.scatter(x, rewards[i])
        plt.xlabel('Iters')
        plt.ylabel('Reward')
        plt.ylim(-5500, -3000)
        plt.text(100, -3250, 'Mean=' + str(round(mean, 1)) + '\nStd=' + str(round(std, 1)), fontdict=font)
        title = 'Final Reward Value with ' + rm_txt
        #plt.title(title)
        plt.savefig(folder + title.replace(' ', '-') + ".png")
        plt.show()

    for i in models:
        rm_txt = 'Reward Modifier - ' + str(to_cmp[i])
        if to_cmp[i] == 1.0:
            rm_txt += ' (No change)'
        plt.scatter(x, ratios[i], label=rm_txt)
    plt.xlabel('Iters')
    plt.ylabel('Undesirable to Total Steps Ratio')
    plt.ylim(0, 0.08)
    title = 'Comparing Final Undesirable Ratio between different Reward Modifiers'
    #plt.title(title)
    plt.savefig(folder + title.replace(' ', '-') + ".png")
    plt.legend()
    plt.show()

    for i in models:

        mean = np.mean(ratios[i])
        std = np.std(ratios[i])

        rm_txt = 'Reward Modifier = ' + str(to_cmp[i])
        if to_cmp[i] == 1.0:
            rm_txt += ' (No change)'

        plt.scatter(x, ratios[i])
        plt.xlabel('Iters')
        plt.ylabel('Undesirable to Total Steps Ratio')
        plt.ylim(0, 0.08)
        plt.text(100, 0.07, 'Mean=' + str(round(mean, 4)) + '\nStd=' + str(round(std, 4)), fontdict=font)
        title = 'Final Undesirable Ratio with ' + rm_txt
        #plt.title(title)
        plt.savefig(folder + title.replace(' ', '-') + ".png")
        plt.show()


def test_file_to_dict(model_num):
    test_file = r"models" + r"\model_" + str(model_num) + r"\test_dicts.txt"

    seeds = []
    rewards = []
    ratios = []
    with open(test_file, "r") as f:
        l = f.readline()
        while l:
            l = l.strip()
            d = json.loads(l)
            seeds.append(d["seed"])
            rewards.append(d["reward"])
            ratios.append(d["ratio"])
            l = f.readline()

    mean_rewards = np.mean(rewards)
    std_rewards = np.std(rewards)
    
    mean_ratios = np.mean(ratios)
    std_ratios = np.std(ratios)

    return mean_rewards, std_rewards, mean_ratios, std_ratios


def get_mean_std_from_models(model_nums):
    mean_rewards = []
    std_rewards = []
    mean_ratios = []
    std_ratios = []

    for model_num in model_nums:
        a,b,c,d = test_file_to_dict(model_num)
        mean_rewards.append(a)
        std_rewards.append(b)
        mean_ratios.append(c)
        std_ratios.append(d)

    return np.mean(mean_rewards), np.std(mean_rewards), np.mean(mean_ratios), np.std(mean_ratios)


def plot_models(modif_to_models):
    """
    Plot the average reward and undesirability of each model trained with each reward modifier
    """
    modifs = []
    mean_rewards = []
    std_rewards = []
    mean_ratios = []
    std_ratios = []
    for modif in modif_to_models.keys():
        modifs.append(modif)
        models = modif_to_models[modif]

        # get mean
        a,b,c,d = get_mean_std_from_models(models)
        mean_rewards.append(a)
        std_rewards.append(b)
        mean_ratios.append(c)
        std_ratios.append(d)

    plt.errorbar(modifs, mean_rewards, std_rewards, linestyle='None', marker='x', capsize=3, ecolor='green')
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(modifs, mean_rewards)

    # use red as color for regression line
    plt.plot(modifs, slope * np.array(modifs) + intercept, color='red', label=f'$R^2={r_value:.3f}$')
    plt.xlabel('Reward Modifier', fontsize=16)
    plt.ylabel('Average reward (Mean and Std)', fontsize=16)
    plt.legend(fontsize=14)

    plt.gcf().subplots_adjust(bottom=0.15)
    plt.gcf().subplots_adjust(top=0.85)
    plt.gcf().subplots_adjust(left=0.15)

    plt.savefig('traffic_reward_comparison.svg')

    plt.show()

    plt.errorbar(modifs, mean_ratios, std_ratios, linestyle='None', marker='x', capsize=3, ecolor='green')
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(modifs, mean_ratios)

    # use red as color for regression line
    plt.plot(modifs, slope * np.array(modifs) + intercept, color='red', label=f'$R^2={r_value:.3f}$')
    plt.xlabel('Reward Modifier', fontsize=16)
    plt.ylabel('Undesirable Behavior Ratio (Mean and Std)',fontsize=16)
    plt.legend(fontsize=14)
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.gcf().subplots_adjust(top=0.85)
    plt.gcf().subplots_adjust(left=0.15)

    plt.savefig('traffic_ratio_comparison.svg')

    plt.show()


def get_time_from_model(run_number):
    """
    Get model training time.
    param: run_number - run number to get the time of
    """
    model = r"models\model_" + str(run_number) + "/times.txt"
    with open(model, "r") as f:
        time_str = f.readline().strip()
        time_str = time_str.split('.')[0]
        hours, minutes, seconds = time_str.split(':')

        # get number of minutes
        num_of_mins = int(hours)*60 + int(minutes) + round(int(seconds)/60)
    return num_of_mins


def get_times_of_group(run_numbers):
    """
    Get time for a run group.
    param: run_numbers - runs to get the time of
    """
    times = []
    for run_number in run_numbers:
        times += [get_time_from_model(run_number)]

    return times

# def print_times_avgs(modifs_dict):
#     for modif in modifs_dict.keys():
#         print(get_times_of_group(modifs_dict[modif]))

def print_times_avgs(modifs_dict):
    """
    Get average of times for each run group.
    param: modifs_dict - the dictionary of {modifier : run_numbers}.
    """
    timess = []
    for modif in modifs_dict.keys():
        timess.append(np.mean(get_times_of_group(modifs_dict[modif])))
    print(timess[0])
    print(np.mean(timess[1:]))

    # the first element in timesss should always be with reward modifier of 1 (regular run).
    print("Original model's time / Model with tree's time: " + str(timess[0] / np.mean(timess[1:])) + "\n")


def get_one_reward_data(run_number):
    """
    Get the reward of a single run.
    param: run_number - the run number.
    """
    model = r"models\model_" + str(run_number) + "/plot_reward_data.txt"
    data = []
    with open(model, "r") as f:
        l = f.readline().strip()
        while l:
            data.append(float(l))
            l = f.readline().strip()
    return data

def multiple_reward(run_numbers):
    """
    Get the average reward from a group of runs.
    param: run_numbers - the group of runs.
    """
    avg = np.array([0.0]*100)

    for run_number in run_numbers:
        avg += get_one_reward_data(run_number)

    avg /= len(run_numbers)

    return avg

def plot_rewards(modif_to_models):
    """
    Plot the average reward for each reward modifier.
    param: modif_to_models - the dictionary of modifier and run_numbers
    """
    for modif in modif_to_models.keys():
        plt.plot(list(range(100)), multiple_reward(modif_to_models[modif]), label=str(modif))
    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.tight_layout()
    plt.legend()
    plt.show()


def get_times():
    """
    Get the time overhead.
    """
    # modif_to_models = {
    #     1: [300, 301, 302, 203],
    #     0.75: [311, 312, 313, 310],
    # }

    # dictionary of {modifier : run_numbers}
    modif_to_models = {
        1: [980, 981, 982, 983, 984, 985, 986, 987, 988, 989],
        0.5: [990, 991, 992, 993, 994, 995, 996, 997, 998, 999]
    }
    print_times_avgs(modif_to_models)

def get_reward_modifier_graphs():
    """
    Get graphs which compare the undesirability per reward modifier.
    """
    # modif_to_models = {
    #     1 : [100, 101, 102, 103],
    #     0.75 : [120, 121, 122, 123],
    #     0.5 : [130, 131, 132, 133],
    #     0.25 : [140, 141, 142, 143],
    #     0.1 : [150, 151, 152, 153]
    # }
    # models to run
    modif_to_models = {
        1: [400, 401, 402],
        0.75: [410, 411, 412],
        0.5: [420, 421, 422],
        0.25: [430, 431, 432],
        0.1: [440, 441, 442],
    }
    # plotting the relevant models
    plot_models(modif_to_models)

def get_models_rewards():
    """
    Plot the model reward graphs.
    """
    # modif_to_models = {
    #     1: [100, 101, 102, 103],
    #     0.75: [120, 121, 122, 123],
    #     0.5: [130, 131, 132, 133],
    #     0.25: [140, 141, 142, 143],
    #     0.1: [150, 151, 152, 153]
    # }

    # the dictionary of modifiers and run_numbers
    modif_to_models = {
        1: [400, 401, 402],
        0.75: [410, 411, 412],
        0.5: [420, 421, 422],
        0.25: [430, 431, 432],
        0.1: [440, 441, 442],
    }
    # plot the rewards
    plot_rewards(modif_to_models)

if __name__ == '__main__':
    while True:
        a = input("Input 1 for the graphs we present in the papers, input 2 for the reward convergence of each model group, input 3 for the time ratio between the original and the new models.\n")
        if int(a) == 1:
            get_reward_modifier_graphs()
        if int(a) == 2:
            get_models_rewards()
        if int(a) == 3:
            get_times()