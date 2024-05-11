import pickle

import numpy as np
from matplotlib import pyplot as plt

from Tree.compare_models import get_big_dict_from_file
import seaborn as sns

def get_models_avgs(run_dict):
    """
    Get averages of models.
    param: run_dict - the runs we want to get the average over.
    """
    new_run_dict = {}

    pth = r"../../models/model_"

    for modif in run_dict.keys():
        rewss = []
        ratss = []
        for run in run_dict[modif]:
            pth_to_run = pth + str(run) + "/test_stats_new.txt"
            d = get_big_dict_from_file(pth_to_run)

            rewss += (d["sum_of_rewards"][:25])
            ratss += (d["Ratio"][:25])

        new_run_dict[modif] = {"rew": rewss, "rat": ratss}

    return new_run_dict


def model_avgs_with_viper(run_dict):
    """
    Get averages of VIPER models.
    """
    new_run_dict = get_models_avgs(run_dict)

    with open("../snake_pre/stud_ratios.pkl", 'rb') as f:
        rats = pickle.load(f)

    with open("../snake_pre/stud_rewards.pkl", 'rb') as f:
        rews = pickle.load(f)

    new_run_dict["VIPER"] = {"rew" : rews, "rat" : rats}

    return new_run_dict


def create_hist(run_dict):
    """
    Create the histogram with the runs.
    param: run_dict - the relevant run.
    """
    new_run_dict = model_avgs_with_viper(run_dict)

    for modif in new_run_dict.keys():
        if modif == 1:
            lab = "Original Model"
        elif modif == 0.1:
            lab = "Our approach (reward modifier = 0.1)"
        else:
            lab = "VIPER"

        sns.distplot(new_run_dict[modif]["rew"], hist=False, label=lab)
    plt.xlabel('Final Reward', size='16')
    plt.ylabel('Probability Density Function', size='16')
    plt.tight_layout()
    plt.legend()
    plt.savefig("pdf_reward_snake.svg")
    plt.show()

    for modif in new_run_dict.keys():
        if modif == 1:
            lab = "Original Model"
        elif modif == 0.1:
            lab = "Our approach (reward modifier = 0.1)"
        else:
            lab = "VIPER"

        print(new_run_dict[modif]["rat"])
        sns.histplot(new_run_dict[modif]["rat"], label=lab)
    plt.xlabel('Undesirable Behavior Ratio', size='16')
    plt.ylabel('Count', size='16')
    plt.tight_layout()
    plt.legend()
    plt.savefig("hist_ratio_snake.svg")
    plt.show()

if __name__ == '__main__':
    # use these runs for comparison between our approach and VIPER
    create_hist({1: {400, 401, 402, 403}, 0.1 : {440, 441, 442, 443}})