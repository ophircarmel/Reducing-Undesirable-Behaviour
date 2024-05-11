from __future__ import absolute_import
from __future__ import print_function

import json
import os
import sys
from shutil import copyfile

from testing_simulation import Simulation
from generator import TrafficGenerator
from model import TestModel
from visualization import Visualization
from utils import import_test_configuration, set_sumo, set_test_path


if __name__ == "__main__":
    # Run and get reward and cntrs for all seeds
    if len(sys.argv) > 1:

        # get test config
        config = import_test_configuration(config_file='testing_settings.ini')
        sumo_cmd = set_sumo(config['gui'], config['sumocfg_file_name'], config['max_steps'])
        model_path, plot_path = set_test_path(config['models_path_name'], config['model_to_test'])

        Model = TestModel(
            input_dim=config['num_states'],
            model_path=model_path
        )

        TrafficGen = TrafficGenerator(
            config['max_steps'],
            config['n_cars_generated']
        )

        # cntr file
        cntr_file = "new_seed_cntr_file.txt"

        last_seed = 10000
        N = 100000

        # get many many seeds
        for i in range(10000, N):
            # get seed
            seed = i
            print('\n----- Test episode ' + str(i) + " out of " + str(N) + ", seed = " + str(seed))

            # create simulation
            sim_seed = Simulation(
                Model,
                TrafficGen,
                sumo_cmd,
                config['max_steps'],
                config['green_duration'],
                config['yellow_duration'],
                config['num_states'],
                config['num_actions']
            )

            # run simulation
            simulation_time = sim_seed.run(seed)  # run the simulation

            # get cntrs
            all_state_cntr = sim_seed.params["all_state_cntr"]
            bad_state_cntr = sim_seed.params["bad_state_cntr"]

            # print time and cntrs
            print('Simulation time:', simulation_time, 's')
            print('States_Cntr: ', all_state_cntr, "\nBad States_Cntr: ", bad_state_cntr)

            # write cntrs to file
            with open(cntr_file, "a+") as f:
                d = {
                    "SEED": seed,
                    "States_Cntr": str(all_state_cntr),
                    "Bad_States_Cntr": str(bad_state_cntr),
                    "Bad_To_All_Ratio": str(round(bad_state_cntr / all_state_cntr, 4))
                }
                json.dump(d, f)
                f.write(os.linesep)

            # reset cntrs
            sim_seed.params["all_state_cntr"] = 0
            sim_seed.params["bad_state_cntr"] = 0

        print("DONE!")
        exit(0)

    ### original file from here - irrelevant
    config = import_test_configuration(config_file='testing_settings.ini')
    sumo_cmd = set_sumo(config['gui'], config['sumocfg_file_name'], config['max_steps'])
    model_path, plot_path = set_test_path(config['models_path_name'], config['model_to_test'])

    Model = TestModel(
        input_dim=config['num_states'],
        model_path=model_path
    )

    TrafficGen = TrafficGenerator(
        config['max_steps'],
        config['n_cars_generated']
    )

    Visualization = Visualization(
        plot_path,
        dpi=96
    )

    Simulation = Simulation(
        Model,
        TrafficGen,
        sumo_cmd,
        config['max_steps'],
        config['green_duration'],
        config['yellow_duration'],
        config['num_states'],
        config['num_actions']
    )

    print('\n----- Test episode')
    simulation_time = Simulation.run(config['episode_seed'])  # run the simulation
    print('Simulation time:', simulation_time, 's')
    print('States_Cntr: ', Simulation.params["all_state_cntr"], "\nBad States_Cntr: ",
          Simulation.params["bad_state_cntr"])

    print("----- Testing info saved at:", plot_path)

    copyfile(src='testing_settings.ini', dst=os.path.join(plot_path, 'testing_settings.ini'))

    Visualization.save_data_and_plot(data=Simulation.reward_episode, filename='reward', xlabel='Action step',
                                     ylabel='Reward')
    Visualization.save_data_and_plot(data=Simulation.queue_length_episode, filename='queue', xlabel='Step',
                                     ylabel='Queue length (vehicles)')
