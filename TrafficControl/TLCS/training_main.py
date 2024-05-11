from __future__ import absolute_import
from __future__ import print_function

import json
import os
import datetime
import sys
from shutil import copyfile
import argparse
import wandb

from training_simulation import Simulation
from generator import TrafficGenerator
from memory import Memory
from model import TrainModel
from visualization import Visualization
from utils import import_train_configuration, set_sumo, set_train_path, set_specific_train_path

if __name__ == "__main__":

    # Construct the argument parser
    ap = argparse.ArgumentParser()

    # Add the arguments to the parser
    ap.add_argument("-ru", "--run_number", required=True,
                    help="Run Number")
    ap.add_argument("-re", "--reward_modifier", required=True,
                    help="Reward Modifier")
    args = vars(ap.parse_args())

    config = import_train_configuration(config_file='training_settings.ini')
    #####################
    ### START ADDITION
    #####################
    config["reward_modifier"] = float(args["reward_modifier"])
    run_number = int(args["run_number"])
    #####################
    ### END ADDITION
    #####################
    sumo_cmd = set_sumo(config['gui'], config['sumocfg_file_name'], config['max_steps'])
    path = set_specific_train_path(config['models_path_name'], str(run_number))

    # create model
    Model = TrainModel(
        config['num_layers'], 
        config['width_layers'], 
        config['batch_size'], 
        config['learning_rate'], 
        input_dim=config['num_states'], 
        output_dim=config['num_actions']
    )

    # memory
    Memory = Memory(
        config['memory_size_max'], 
        config['memory_size_min']
    )

    # traffic generator
    TrafficGen = TrafficGenerator(
        config['max_steps'], 
        config['n_cars_generated']
    )

    Visualization = Visualization(
        path, 
        dpi=96
    )

    # create simulation
    Simulation = Simulation(
        Model,
        Memory,
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

    # wandb.init(
    #     project="Traffic_Control",
    #     entity = "enter_entity_name_here",
    #     name='Run  ' + str(run_number) + ' - Modifier = ' + str(config['reward_modifier']),
    #     save_code=True,
    #     tags=[str(config['reward_modifier'])],
    #     id='run' + str(run_number),
    #     config=config
    # )
    
    while episode < config['total_episodes']:
        print('\n----- Episode', str(episode+1), 'of', str(config['total_episodes']))
        epsilon = 1.0 - (episode / config['total_episodes'])  # set the epsilon for this episode according to epsilon-greedy policy
        simulation_time, training_time = Simulation.run(episode, epsilon)  # run the simulation

        ###################
        # Start addition
        ####################

        cntr_file = "train_cntr.txt"

        all_state_cntr = Simulation.params["all_state_cntr"]
        bad_state_cntr = Simulation.params["bad_state_cntr"]

        # print time and cntrs
        print('States_Cntr: ', all_state_cntr, "\nBad States_Cntr: ", bad_state_cntr)

        # # write cntrs to file
        # with open(cntr_file, "a+") as f:
        #     d = {
        #         "Episode": episode,
        #         "States_Cntr": str(all_state_cntr),
        #         "Bad_States_Cntr": str(bad_state_cntr),
        #         "Bad_To_All_Ratio": str(round(bad_state_cntr / (all_state_cntr+0.001), 4))
        #     }
        #     json.dump(d, f)
        #     f.write(os.linesep)

        # if all_state_cntr != 0:
        #     wandb.log(
        #         {
        #             "Episode" : episode,
        #             "Simulation Time" : simulation_time,
        #             "Training Time" : training_time,
        #             "Reward " : Simulation.reward_store[-1],
        #             "Undesirable to Total Ratio" : round(bad_state_cntr / (all_state_cntr), 4)
        #         }
        #     )



        # reset cntrs
        Simulation.params["all_state_cntr"] = 0
        Simulation.params["bad_state_cntr"] = 0

        #####################
        # END ADDITION
        #####################

        print('Simulation time:', simulation_time, 's - Training time:', training_time, 's - Total:', round(simulation_time+training_time, 1), 's')
        episode += 1

    #####################
    # START ADDITION
    #####################
    # handle time
    timestamp_end = datetime.datetime.now()

    print("\n----- Start time:", timestamp_start)
    print("----- End time:", datetime.datetime.now())

    with open(path + "/times.txt", "w+") as f:
        f.write(str(timestamp_end - timestamp_start))

    #####################
    # END ADDITION
    #####################

    print("----- Session info saved at:", path)

    Model.save_model(path)

    copyfile(src='training_settings.ini', dst=os.path.join(path, 'training_settings.ini'))

    Visualization.save_data_and_plot(data=Simulation.reward_store, filename='reward', xlabel='Episode', ylabel='Cumulative negative reward')
    Visualization.save_data_and_plot(data=Simulation.cumulative_wait_store, filename='delay', xlabel='Episode', ylabel='Cumulative delay (s)')
    Visualization.save_data_and_plot(data=Simulation.avg_queue_length_store, filename='queue', xlabel='Episode', ylabel='Average queue length (vehicles)')