from __future__ import absolute_import
from __future__ import print_function

import json
import os
import datetime
import sys
from shutil import copyfile

import wandb

from training_simulation import Simulation
from generator import TrafficGenerator
from memory import Memory
from model import TrainModel
from visualization import Visualization
from utils import import_train_configuration, set_sumo, set_train_path

def train_one(config):
    sumo_cmd = set_sumo(config['gui'], config['sumocfg_file_name'], config['max_steps'])
    path = set_train_path(config['models_path_name'])

    with open(path + "\config.ini", "w+") as f:
        json.dump(config, f)

    Model = TrainModel(
        config['num_layers'],
        config['width_layers'],
        config['batch_size'],
        config['learning_rate'],
        input_dim=config['num_states'],
        output_dim=config['num_actions']
    )

    memory = Memory(
        config['memory_size_max'],
        config['memory_size_min']
    )

    TrafficGen = TrafficGenerator(
        config['max_steps'],
        config['n_cars_generated']
    )

    visualization = Visualization(
        path,
        dpi=96
    )

    simulation = Simulation(
        Model,
        memory,
        TrafficGen,
        sumo_cmd,
        config['gamma'],
        config['max_steps'],
        config['green_duration'],
        config['yellow_duration'],
        config['num_states'],
        config['num_actions'],
        config['training_epochs'],
        config['reward_modifier']
    )

    episode = 0
    timestamp_start = datetime.datetime.now()

    run_number = int((path[-5:].split('_'))[1].split('\\')[0])

    wandb.init(
    project = "Traffic_Control",
    entity = "enter_entity_name_here",
    name = 'Run  ' + str(run_number) + ' - Modifier = ' + str(['reward_modifier']),
    save_code = True,
    tags = [str(['reward_modifier'])],
    id = 'run' + str(run_number),
    config = config
    )

    while episode < config['total_episodes']:
        print('\n----- Episode', str(episode + 1), 'of', str(config['total_episodes']))
        epsilon = 1.0 - (episode / config[
            'total_episodes'])  # set the epsilon for this episode according to epsilon-greedy policy
        simulation_time, training_time = simulation.run(episode, epsilon)  # run the simulation

        ###################
        # Start addition
        ####################

        cntr_file = "train_cntr.txt"

        all_state_cntr = simulation.params["all_state_cntr"]
        bad_state_cntr = simulation.params["bad_state_cntr"]

        # print time and cntrs
        print('States_Cntr: ', all_state_cntr, "\nBad States_Cntr: ", bad_state_cntr)

        # write cntrs to file
        # with open(cntr_file, "a+") as f:
        #     d = {
        #         "Episode": episode,
        #         "States_Cntr": str(all_state_cntr),
        #         "Bad_States_Cntr": str(bad_state_cntr),
        #         "Bad_To_All_Ratio": str(round(bad_state_cntr / (all_state_cntr + 0.001), 4))
        #     }
        #     json.dump(d, f)
        #     f.write(os.linesep)

        if all_state_cntr != 0:
            wandb.log(
                {
                    "Episode" : episode,
                    "Simulation Time" : simulation_time,
                    "Training Time" : training_time,
                    "Reward " : simulation.reward_store[-1],
                    "Undesirable to Total Ratio" : (round(bad_state_cntr / (all_state_cntr), 4))
                }
            )


        # reset cntrs
        simulation.params["all_state_cntr"] = 0
        simulation.params["bad_state_cntr"] = 0

        #####################
        # END ADDITION
        #####################

        print('Simulation time:', simulation_time, 's - Training time:', training_time, 's - Total:',
              round(simulation_time + training_time, 1), 's')
        episode += 1

    print("\n----- Start time:", timestamp_start)
    print("----- End time:", datetime.datetime.now())
    print("----- Session info saved at:", path)

    Model.save_model(path)

    #copyfile(src='training_settings.ini', dst=os.path.join(path, 'training_settings.ini'))

    visualization.save_data_and_plot(data=simulation.reward_store, filename='reward', xlabel='Episode',
                                     ylabel='Cumulative negative reward')
    visualization.save_data_and_plot(data=simulation.cumulative_wait_store, filename='delay', xlabel='Episode',
                                     ylabel='Cumulative delay (s)')
    visualization.save_data_and_plot(data=simulation.avg_queue_length_store, filename='queue', xlabel='Episode',
                                     ylabel='Average queue length (vehicles)')

def main():
    args = sys.argv[1:]
    modifiers = [float(arg) for arg in args]

    for modifier in modifiers:
        print(modifier)
        config = import_train_configuration(config_file='training_settings.ini')

        config["reward_modifier"] = modifier
        # to_modify = json.loads(args[i])
        # for key in to_modify:
        #     config[key] = to_modify[key]

        train_one(config)



if __name__ == "__main__":
    main()