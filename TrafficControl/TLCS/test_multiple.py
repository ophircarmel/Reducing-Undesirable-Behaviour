from __future__ import absolute_import
from __future__ import print_function

import json
import os
import pickle
import sys
from shutil import copyfile

import numpy as np
from matplotlib import pyplot as plt

from compare_models import get_mean_std_from_models, test_file_to_dict
from testing_simulation import Simulation
from generator import TrafficGenerator
from model import TestModel
from viper_TrafficControl.pong.main import cnt_undesirable
from visualization import Visualization
from utils import import_test_configuration, set_sumo, set_test_path

# you may need to change this path to the absolute path
PTH = r"TrafficControl\TLCS\models/"
#PTH = r"models/"


def compare_models_with_rewards(model_nums):
    config = import_test_configuration(config_file='testing_settings.ini')
    sumo_cmd = set_sumo(config['gui'], config['sumocfg_file_name'], config['max_steps'])

    TrafficGen = TrafficGenerator(
        config['max_steps'],
        config['n_cars_generated']
    )

    # cntr file
    cntr_file = \
        "compare_models_" \
        + "-".join([str(mod_num) for mod_num in model_nums]) \
        + ".txt"

    initial_seed = 0#config['init_seed']
    N = 150
    rewards = {}
    ratios = {}
    seeds = {}
    
    for model_num in model_nums:
        rewards[model_num] = []
        ratios[model_num] = []
        seeds[model_num] = []

    # get many many seeds
    for i in range(initial_seed, N):
        # get seed
        seed = i
        print("Seed = " + str(seed))

        for model_num in model_nums:
            # get the path and the plot path
            model_path, plot_path = set_test_path(config['models_path_name'], model_num)
            new_plot_path = plot_path + "seed_" + str(seed) + "\\"
            if os.path.exists(new_plot_path):
                import shutil
                shutil.rmtree(new_plot_path)
            os.mkdir(new_plot_path)

            mod = TestModel(
                input_dim=config['num_states'],
                model_path=model_path
            )

            # create simulation
            sim = Simulation(
                mod,
                TrafficGen,
                sumo_cmd,
                config['max_steps'],
                config['green_duration'],
                config['yellow_duration'],
                config['num_states'],
                config['num_actions']
            )

            # run simulation
            simulation_time, reward = sim.run(seed)  # run the simulation

            # get cntrs
            all_state_cntr = sim.params["all_state_cntr"]
            bad_state_cntr = sim.params["bad_state_cntr"]
            ratio = round(bad_state_cntr / all_state_cntr, 4)

            rewards[model_num].append(reward)
            ratios[model_num].append(ratio)
            seeds[model_num].append(seed)
            
            if (i - initial_seed) % 10 == 0:
                plt.scatter(list(range(len(rewards[model_num]))), rewards[model_num],
                            label='Reward: model_num=' + str(model_num))
            if (i - initial_seed) % 10  == 5:
                plt.scatter(list(range(len(ratios[model_num]))), ratios[model_num],
                            label='Ratio: model_num=' + str(model_num))

            # print time and cntrs
            print("Model #: " + str(model_num))
            print("Iteration Num: " + str(i - initial_seed))
            print('Simulation time:', simulation_time, 's')
            print('States_Cntr: ', all_state_cntr, "\nBad States_Cntr: ", bad_state_cntr)
            print('Ratio: ', ratio)
            print("Reward: ", reward)
            print("\n")
            print('Mean of Rewards of Model ' + str(model_num) + ' = ' + str(round(np.mean(rewards[model_num]), 2)))
            print('STD of Rewards of Model ' + str(model_num) + ' = ' + str(round(np.std(rewards[model_num]), 2)))
            print('Mean of Ratios of Model ' + str(model_num) + ' = ' + str(round(np.mean(ratios[model_num]), 2)))
            print('STD of Ratios of Model ' + str(model_num) + ' = ' + str(round(np.std(ratios[model_num]), 2)))
            print("\n\n")

            
            vis = Visualization(
                new_plot_path,
                dpi=96
            )

            copyfile(src='testing_settings.ini', dst=os.path.join(new_plot_path, 'testing_settings.ini'))

            vis.save_data_and_plot(data=sim.reward_episode, filename='reward', xlabel='Action step',
                                   ylabel='Reward')
            vis.save_data_and_plot(data=sim.queue_length_episode, filename='queue', xlabel='Step',
                                   ylabel='Queue length (vehicles)')
        plt.title('Comparative')
        plt.legend()
        plt.show()

        print(rewards)
        print(ratios)

        with open(cntr_file, 'wb+') as f:
            d = {'rewards': rewards, 'ratios': ratios, 'seeds' : seeds}
            pickle.dump(d, f)

def compare_models(model_nums):
    config = import_test_configuration(config_file='testing_settings.ini')
    sumo_cmd = set_sumo(config['gui'], config['sumocfg_file_name'], config['max_steps'])

    TrafficGen = TrafficGenerator(
        config['max_steps'],
        config['n_cars_generated']
    )

    # cntr file
    cntr_file = \
        "compare_models_by_seed_" \
        + "-".join([str(mod_num) for mod_num in model_nums])\
        + ".txt"

    initial_seed = config['init_seed']
    N = 100000

    # get many many seeds
    for i in range(initial_seed, N):
        # get seed
        seed = i
        print("Seed = " + str(seed))

        seed_dict = {}

        for model_num in model_nums:
            # get the path and the plot path
            model_path, plot_path = set_test_path(config['models_path_name'], model_num)
            new_plot_path = plot_path + "seed_" + str(seed) + "\\"
            if os.path.exists(new_plot_path):
                import shutil
                shutil.rmtree(new_plot_path)
            os.mkdir(new_plot_path)


            mod = TestModel(
                input_dim=config['num_states'],
                model_path=model_path
            )

            # create simulation
            sim = Simulation(
                mod,
                TrafficGen,
                sumo_cmd,
                config['max_steps'],
                config['green_duration'],
                config['yellow_duration'],
                config['num_states'],
                config['num_actions']
            )

            # run simulation
            simulation_time = sim.run(seed)  # run the simulation

            # get cntrs
            all_state_cntr = sim.params["all_state_cntr"]
            bad_state_cntr = sim.params["bad_state_cntr"]
            ratio = round(bad_state_cntr / all_state_cntr, 4)


            # print time and cntrs
            print("Model #: " + str(model_num))
            print('Simulation time:', simulation_time, 's')
            print('States_Cntr: ', all_state_cntr, "\nBad States_Cntr: ", bad_state_cntr)
            print('Ratio: ', ratio)

            print("\n\n")

            # put in dictionary
            seed_dict['Model ' + str(model_num)] = ratio


            vis = Visualization(
                new_plot_path,
                dpi=96
            )

            copyfile(src='testing_settings.ini', dst=os.path.join(new_plot_path, 'testing_settings.ini'))

            vis.save_data_and_plot(data=sim.reward_episode, filename='reward', xlabel='Action step',
                                             ylabel='Reward')
            vis.save_data_and_plot(data=sim.queue_length_episode, filename='queue', xlabel='Step',
                                             ylabel='Queue length (vehicles)')

        # write cntrs to file
        with open(cntr_file, "a+") as f:
            json.dump(seed_dict, f)
            f.write(os.linesep)

def test_single_model(model_num):
    """
    Test one model. Get the undesirable behavior ratio nad average reward.
    param: model_num - the run_number we want to test
    """
    # path to model
    path_to_model = PTH + "model_" + str(model_num) + "/"

    # configurrations
    config = import_test_configuration(config_file='testing_settings.ini')
    sumo_cmd = set_sumo(config['gui'], config['sumocfg_file_name'], config['max_steps'])

    model_path, plot_path = set_test_path(config['models_path_name'], model_num)


    # create traffic generator
    TrafficGen = TrafficGenerator(
        config['max_steps'],
        config['n_cars_generated']
    )

    initial_seed = 0  # config['init_seed']
    N = 100
    rewards = []
    ratios = []
    seeds = []

    # get many many seeds
    for i in range(initial_seed, N):
        # get seed
        seed = i
        print("Seed = " + str(seed))

        mod = TestModel(
            input_dim=config['num_states'],
            model_path=model_path
        )

        # create simulation
        sim = Simulation(
            mod,
            TrafficGen,
            sumo_cmd,
            config['max_steps'],
            config['green_duration'],
            config['yellow_duration'],
            config['num_states'],
            config['num_actions']
        )

        # run simulation
        simulation_time, reward = sim.run(seed)  # run the simulation

        # get cntrs
        all_state_cntr = sim.params["all_state_cntr"]
        bad_state_cntr = sim.params["bad_state_cntr"]
        ratio = round(bad_state_cntr / all_state_cntr, 4)

        rewards.append(reward)
        ratios.append(ratio)

        # print time and cntrs
        print("Iteration Num: " + str(i - initial_seed))
        print('Simulation time:', simulation_time, 's')
        print('States_Cntr: ', all_state_cntr, "\nBad States_Cntr: ", bad_state_cntr)
        print('Ratio: ', ratio)
        print("Reward: ", reward)
        print("\n")
        print('Mean of Rewards of Model ' + str(model_num) + ' = ' + str(round(np.mean(rewards), 2)))
        print('STD of Rewards of Model ' + str(model_num) + ' = ' + str(round(np.std(rewards), 2)))
        print('Mean of Ratios of Model ' + str(model_num) + ' = ' + str(round(np.mean(ratios), 2)))
        print('STD of Ratios of Model ' + str(model_num) + ' = ' + str(round(np.std(ratios), 2)))
        print("\n\n")

        new_plot_path = plot_path + "seed_" + str(seed) + "\\"
        if os.path.exists(new_plot_path):
            import shutil
            shutil.rmtree(new_plot_path)
        os.mkdir(new_plot_path)

        vis = Visualization(
            new_plot_path,
            dpi=96
        )

        copyfile(src='testing_settings.ini', dst=os.path.join(new_plot_path, 'testing_settings.ini'))

        # visualize
        vis.save_data_and_plot(data=sim.reward_episode, filename='reward', xlabel='Action step',
                               ylabel='Reward')
        vis.save_data_and_plot(data=sim.queue_length_episode, filename='queue', xlabel='Step',
                               ylabel='Queue length (vehicles)')

        # save dictionary of the seed, and the reward and undesirable ratio.
        dict_to_save = {
            'seed' : i,
            'reward' : rewards[-1],
            'ratio' : ratios[-1]
        }

        dict_file = path_to_model + "test_dicts.txt"

        # save dictionary
        with open(dict_file, 'a+') as f:
            f.write(json.dumps(dict_to_save))
            f.write('\n')

    print(rewards)
    print(ratios)

def test_multiple_models_and_viper(modif_to_models, N):

    import seaborn as sns

    modifs = []
    for modif in modif_to_models.keys():
        modifs.append(modif)
        models = modif_to_models[modif]

        mean_rew = []
        mean_rat = []

        for model_num in models:
            test_file = r"models" + r"\model_" + str(
                model_num) + r"\test_dicts.txt"

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

            if mean_rew == []:
                mean_rew = np.array(rewards)
            else:
                mean_rew += np.array(rewards)

            if mean_rat == []:
                mean_rat = np.array(ratios)
            else:
                mean_rat += np.array(ratios)

        lab = ""

        if modif == 1:
            lab = "Original Model"
        else:
            lab = "Our approach (reward modifier = 0.1)"

        sns.distplot(mean_rew / len(models), hist=False, label=lab)

    stud_ratios, stud_rewards = cnt_undesirable(N)

    sns.distplot(stud_rewards, hist=False, label='VIPER')
    plt.xlabel('Final Reward', size='16')
    plt.ylabel('Probability Density Function', size='16')
    plt.tight_layout()
    #plt.title('Density Plot Comparison of Undesirable Behavior Ratios')
    plt.legend()


    plt.savefig("pdf_reward_traffic.svg")
    plt.show()

    modifs = []
    for modif in modif_to_models.keys():
        modifs.append(modif)
        models = modif_to_models[modif]

        mean_rew = []
        mean_rat = []

        for model_num in models:
            # you may need to change this path to the absolute path
            test_file = r"TrafficControl\TLCS\models" + r"\model_" + str(
                model_num) + r"\test_dicts.txt"

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

            if mean_rew == []:
                mean_rew = np.array(rewards)
            else:
                mean_rew += np.array(rewards)

            if mean_rat == []:
                mean_rat = np.array(ratios)
            else:
                mean_rat += np.array(ratios)

        lab = ""

        if modif == 1:
            lab = "Original Model"
        else:
            lab = "Our approach (reward modifier = 0.1)"

        sns.histplot(mean_rat / len(models), label=lab)

    sns.histplot(stud_ratios, label='VIPER')
    plt.xlabel('Undesirable Behavior Ratio', size='16')
    plt.ylabel('Count', size='16')
    plt.tight_layout()
    #plt.title('Density Plot Comparison of Undesirable Behavior Ratios')
    plt.legend()


    plt.savefig("hist_ratio_traffic.svg")
    plt.show()






def test_multiple_models_and_student(modelss, N):
    stud_ratios, stud_rewards = cnt_undesirable(N)
    print(stud_rewards)

    ratiosss = []
    rewardsss = []
    for models in modelss:
        ratioss = []
        rewardss = []
        for model in models:
            config = import_test_configuration(config_file='testing_settings.ini')
            sumo_cmd = set_sumo(config['gui'], config['sumocfg_file_name'], config['max_steps'])

            model_path, plot_path = set_test_path(config['models_path_name'], model)

            TrafficGen = TrafficGenerator(
                config['max_steps'],
                config['n_cars_generated']
            )

            initial_seed = 0  # config['init_seed']
            rewards = []
            ratios = []

            mod = TestModel(
                input_dim=config['num_states'],
                model_path=model_path
            )

            # get many many seeds
            for i in range(0, N):
                # get seed
                seed = i
                print("Seed = " + str(seed))

                # create simulation
                sim = Simulation(
                    mod,
                    TrafficGen,
                    sumo_cmd,
                    config['max_steps'],
                    config['green_duration'],
                    config['yellow_duration'],
                    config['num_states'],
                    config['num_actions']
                )

                # run simulation
                simulation_time, reward = sim.run(seed)  # run the simulation

                # get cntrs
                all_state_cntr = sim.params["all_state_cntr"]
                bad_state_cntr = sim.params["bad_state_cntr"]
                ratio = round(bad_state_cntr / all_state_cntr, 4)

                rewards.append(reward)
                ratios.append(ratio)

                # print time and cntrs
                print("Iteration Num: " + str(i - initial_seed))
                print('Simulation time:', simulation_time, 's')
                print('States_Cntr: ', all_state_cntr, "\nBad States_Cntr: ", bad_state_cntr)
                print('Ratio: ', ratio)
                print("Reward: ", reward)
                print("\n")
                print('Mean of Rewards of Model ' + str(model) + ' = ' + str(round(np.mean(rewards), 2)))
                print('STD of Rewards of Model ' + str(model) + ' = ' + str(round(np.std(rewards), 2)))
                print('Mean of Ratios of Model ' + str(model) + ' = ' + str(round(np.mean(ratios), 2)))
                print('STD of Ratios of Model ' + str(model) + ' = ' + str(round(np.std(ratios), 2)))
                print("\n\n")

            if rewardss != []:
                rewardss += np.array(rewards)
            else:
                rewardss = np.array(rewards)

            if ratioss != []:
                ratioss += np.array(ratios)
            else:
                ratioss = np.array(ratios)
        ratiosss.append(ratioss / len(ratioss))
        rewardsss.append(rewardss / len(rewardss))

    with open("ratios-" + str(N) + ".pkl", 'wb+') as f:
        pickle.dump(ratiosss, f)

    with open("rewards-" + str(N) + ".pkl", 'wb+') as f:
        pickle.dump(rewardsss, f)

    stud_ratios, stud_rewards = cnt_undesirable(N)


    ratiosss.append(np.array(stud_ratios))
    rewardsss.append(np.array(stud_rewards))

    for i in range(len(ratiosss)):
        plt.hist(ratiosss[i], label=str(i))

    plt.legend()
    plt.show()

    for i in range(len(rewardsss)):
        plt.hist(rewardsss[i], label=str(i))

    plt.legend()
    plt.show()




if __name__ == "__main__":
    # get list of run_numbers from args
    models_to_compare = [int(model_num) for model_num in sys.argv[1:]]
    #compare_models_with_rewards(models_to_compare)

    # if wanting to compare with VIPER - configure flag to be 'VIPER'
    flag = None#'VIPER'

    if flag == 'VIPER':
        # compare these specific models with VIPER
        test_multiple_models_and_viper({1: [400, 401, 402], 0.1 : [440, 441, 442]}, 100)

    else:
        # test on single run number
        test_single_model(models_to_compare[0])