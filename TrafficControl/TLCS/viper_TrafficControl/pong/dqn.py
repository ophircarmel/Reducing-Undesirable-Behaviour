# Copyright 2017-2018 MIT
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys

import numpy as np
import tensorflow as tf
import traci

from TrafficControl.TLCS.generator import TrafficGenerator
from TrafficControl.TLCS.model import TestModel
from TrafficControl.TLCS.utils import import_test_configuration, set_sumo, set_test_path
from TrafficControl.TLCS.testing_simulation import Simulation

# you may need to change this path to the absolute path
PTH = r"TrafficControl\TLCS\\"
# you may need to change this path to the absolute path
PTH_TO_SETTINGS = r"TrafficControl\TLCS\testing_settings.ini"


# As described in https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
# Fitted to work with Traffic-Control
class DQNPolicy:
    def __init__(self, model_num):
        #####################
        ### START ADDITION
        #####################
        # initialize simulation
        self.config = import_test_configuration(config_file=PTH_TO_SETTINGS)
        self.sumo_cmd = set_sumo(self.config['gui'], self.config['sumocfg_file_name'], self.config['max_steps'])
        self.sumo_cmd[2] = PTH + self.sumo_cmd[2]

        # self.model_path, self.plot_path = set_test_path(self.config['models_path_name'], model_num)

        model_folder_path = PTH + "models\\model_" + str(model_num)

        if os.path.isdir(model_folder_path):
            plot_path = os.path.join(model_folder_path, 'test', '')
            os.makedirs(os.path.dirname(plot_path), exist_ok=True)
            self.model_path = model_folder_path
            self.plot_path = plot_path
        else:
            sys.exit('The model number specified does not exist in the models folder')

        self.TrafficGen = TrafficGenerator(
            self.config['max_steps'],
            self.config['n_cars_generated']
        )

        self.mod = TestModel(
            input_dim=self.config['num_states'],
            model_path=self.model_path)

        # create simulation
        self.sim = Simulation(
            self.mod,
            self.TrafficGen,
            self.sumo_cmd,
            self.config['max_steps'],
            self.config['green_duration'],
            self.config['yellow_duration'],
            self.config['num_states'],
            self.config['num_actions']
        )
        #####################
        ### END ADDITION
        #####################


    def reset(self):
        #####################
        ### START ADDITION
        #####################
        # reset simulation
        self.sim = Simulation(
            self.mod,
            self.TrafficGen,
            self.sumo_cmd,
            self.config['max_steps'],
            self.config['green_duration'],
            self.config['yellow_duration'],
            self.config['num_states'],
            self.config['num_actions']
        )
        #####################
        ### END ADDITION
        #####################

    def predict_q(self, states):
        qs = self.mod.predict_batch(states)
        return qs

    def predict(self, states):
        # will be problematic
        acts = []
        for state in states:
            acts.append(np.argmax(self.mod.predict_one(state)))

        return acts

    def get_rollout_teacher(self, ep):
        """
        The original rollout_teacher function fitted to work with the Traffic-Control case study
        """
        self.reset()
        #####################
        ### START ADDITION
        #####################
        self.sim._TrafficGen.generate_routefile(seed=ep)
        traci.start(self.sim._sumo_cmd)


        obs, done = np.array(self.sim._get_state()), False
        rollout = []


        self.sim._waiting_times = {}
        steps = 0
        old_total_wait = 0
        old_action = -1

        while not done:

            # get current state of the intersection
            current_state = self.sim._get_state()

            # calculate reward of previous action: (change in cumulative waiting time between actions)
            # waiting time = seconds waited by a car since the spawn in the environment, cumulated for every car in incoming lanes
            current_total_wait = self.sim._collect_waiting_times()
            reward = old_total_wait - current_total_wait

            if self.sim._step != 0:
                rollout.append((old_state, old_action, reward))

            # choose the light phase to activate, based on the current state of the intersection
            action = self.sim._choose_action(current_state)

            # if the chosen phase is different from the last phase, activate the yellow phase
            if self.sim._step != 0 and old_action != action:
                self.sim._set_yellow_phase(old_action)
                self.sim._simulate(self.sim._yellow_duration)

            # execute the phase selected before
            self.sim._set_green_phase(action)
            self.sim._simulate(self.sim._green_duration)

            # saving variables for later & accumulate reward
            old_action = action
            old_state = current_state
            old_total_wait = current_total_wait

            steps += 1

            done = (self.sim._step >= self.sim._max_steps)

        traci.close()
        #####################
        ### END ADDITION
        #####################
        return rollout

    def get_rollouts(self, n_batch_rollouts):
        rollouts = []
        for i in range(n_batch_rollouts):
            print(i)
            rollouts.extend(self.get_rollout_teacher(i))
        return rollouts

