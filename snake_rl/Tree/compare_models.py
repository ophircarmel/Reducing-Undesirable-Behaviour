import json

import matplotlib.pyplot as plt
import numpy as np
import scipy


def get_big_dict_from_file(file_pth):
    with open(file_pth, "r") as f:
        l = f.readline()
        start_d = json.loads(l)

    return start_d


def get_avg_of_lists_in_dict(d):
    new_d = {}
    for key in d.keys():
        new_d[key] = (np.mean(d[key]), np.std(d[key]))

    return new_d


def compare_runs_with_same_modif(runs, pth):

    avgs = {}
    stds = {}
    for run in runs:
        pth_to_run = pth + str(run) + "/test_stats_new.txt"

        avgs_and_stds = get_avg_of_lists_in_dict(get_big_dict_from_file(pth_to_run))

        if avgs == {}:
            for key in avgs_and_stds.keys():
                avgs[key] = [avgs_and_stds[key][0]]
                stds[key] = [avgs_and_stds[key][1]]

        else:
            for key in avgs_and_stds.keys():
                avgs[key].append(avgs_and_stds[key][0])
                stds[key].append(avgs_and_stds[key][1])


    return get_avg_of_lists_in_dict(avgs), stds


def compare_avgs_of_runs(run_dict):
    tot_avg = {}
    tot_std = {}
    for i, modif in enumerate(run_dict):
        print(modif)
        (avg, _) = compare_runs_with_same_modif(run_dict[modif], pth=r"..\models\model_")

        for key in (list(avg.keys())):
            if i == 0:
                tot_avg[key] = [avg[key][0]]
                tot_std[key] = [avg[key][1]]
            else:
                tot_avg[key].append(avg[key][0])
                tot_std[key].append(avg[key][1])


    modifs = list(run_dict.keys())
    mean_rewards = tot_avg["sum_of_rewards"]
    std_rewards = tot_std["sum_of_rewards"]
    mean_ratios = tot_avg["Ratio"]
    std_ratios = tot_std["Ratio"]

    plt.errorbar(modifs, mean_rewards, std_rewards, linestyle='None', marker='x', capsize=3, ecolor='green')
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(modifs, mean_rewards)

    # use red as color for regression line
    plt.plot(modifs, slope * np.array(modifs) + intercept, color='red', label=f'$R^2={r_value*r_value:.3f}$')
    plt.xlabel('Reward Modifier', fontsize=16)
    plt.ylabel('Average reward (Mean and Std)', fontsize=16)
    plt.legend(fontsize=14)
    plt.savefig('snake_reward_new.svg')
    plt.show()

    plt.errorbar(modifs, mean_ratios, std_ratios, linestyle='None', marker='x', capsize=3, ecolor='green')
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(modifs, mean_ratios)

    # use red as color for regression line
    plt.plot(modifs, slope * np.array(modifs) + intercept, color='red', label=f'$R^2={r_value*r_value:.3f}$')
    plt.xlabel('Reward Modifier', fontsize=16)
    plt.ylabel('Undesirable Behavior Ratio (Mean and Std)', fontsize=16)
    plt.legend(fontsize=14)
    plt.savefig('snake_ratio_new.svg')
    plt.show()

def plot_stats(runs_dict):
    modifs = []
    avg_modifs = []
    for modif in runs_dict.keys():
        modifs.append(modif)
        avg_modif = get_stats_of_group(runs_dict[modif])
        plt.plot(avg_modif, label=str(modif))

    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.tight_layout()
    plt.legend()
    plt.show()


def get_stats_of_group(run_numbers):
    avg = [0.0]*50
    avg = np.array(avg)
    for run_number in run_numbers:
        avg += get_stats(run_number)

    return avg/len(run_numbers)

def get_stats(run_number):
    file = r"..\models\model_" + str(run_number) + "/stats.txt"
    with open(file, "r") as f:
        j = json.loads(f.readline().strip())

    sum_rewards = j["sum_of_rewards"]

    return sum_rewards

def get_time_from_model(run_number):
    model = r"..\models\model_" + str(run_number) + "/time.txt"
    with open(model, "r") as f:
        time_str = f.readline().strip()
        time_str = time_str.split('.')[0]
        hours, minutes, seconds = time_str.split(':')

        num_of_mins = int(hours)*60 + int(minutes) + round(int(seconds)/60)
    return num_of_mins


def get_times_of_group(run_numbers):
    times = []
    for run_number in run_numbers:
        times += [get_time_from_model(run_number)]

    return times

def print_times_avgs():
    modifs_dict = {1: [900, 901, 902, 903, 904], 0.1 : [910, 911, 912, 913, 914]}
    timess = []
    for modif in modifs_dict.keys():
        timess.append(np.mean(get_times_of_group(modifs_dict[modif])))
    print(timess[0])
    print(np.mean(timess[1:]))
    print("Original model's time / Model with tree's time: " + str(timess[0] / np.mean(timess[1:])) + "\n")



if __name__ == '__main__':
    # runs_dict = {
    #     1 : [100, 101, 102, 103],
    #     0.75 : [111, 112, 113],
    #     0.5 : [120, 121, 122, 123],
    #     0.25 : [130, 131, 132, 133],
    #     0.1 : [140, 141, 142, 143]
    # }

    # runs_dict = {
    #     1: [200, 201, 202, 203],
    #     0.75: [211, 212, 213],
    #     0.5: [220, 221, 222, 223],
    #     0.25: [230, 231, 232, 233],
    #     0.1: [240, 241, 242]
    # }
    #
    # runs_dict = {
    #     1: [300, 301, 302, 303],
    #     0.75: [310, 311, 312, 313],
    #     0.5: [320, 321, 322, 323],
    #     0.25: [330, 331, 332, 333],
    #     0.1: [340, 341, 342, 342]
    # }

    runs_dict = {
        1: [400, 401, 402, 403],
        0.75: [410, 411, 412, 413],
        0.5: [420, 421, 422, 423],
        0.25: [430, 431, 432, 433],
        0.1: [440, 441, 442, 443]
    }



    while True:
        a = input(
            "Input 1 for the graphs we present in the papers, input 2 for the reward convergence of each model group, input 3 for the time ratio between the original and the new models.\n")
        if int(a) == 1:
            compare_avgs_of_runs(runs_dict)
        if int(a) == 2:
            plot_stats(runs_dict)
        if int(a) == 3:
            print_times_avgs()

