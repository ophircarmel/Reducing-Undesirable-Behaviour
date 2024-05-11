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
import pickle as pk
import numpy as np
import traci
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz

from TrafficControl.TLCS.generator import TrafficGenerator
from TrafficControl.TLCS.testing_simulation import Simulation
from TrafficControl.TLCS.utils import import_test_configuration, set_sumo
from ..pong.dqn import PTH
from ..util.log import *

from TrafficControl.TLCS.viper_TrafficControl.pong.dqn import PTH_TO_SETTINGS


def accuracy(policy, obss, acts):
    return np.mean(acts == policy.predict(obss))


def split_train_test(obss, acts, train_frac):
    n_train = int(train_frac * len(obss))
    idx = np.arange(len(obss))
    np.random.shuffle(idx)
    obss_train = obss[idx[:n_train]]
    acts_train = acts[idx[:n_train]]
    obss_test = obss[idx[n_train:]]
    acts_test = acts[idx[n_train:]]
    return obss_train, acts_train, obss_test, acts_test


def save_dt_policy(dt_policy, dirname, fname):
    if not os.path.isdir(dirname):
        os.makedirs(dirname)
    f = open(dirname + '/' + fname, 'wb')
    pk.dump(dt_policy, f)
    f.close()


def save_dt_policy_viz(dt_policy, dirname, fname):
    if not os.path.isdir(dirname):
        os.makedirs(dirname)
    export_graphviz(dt_policy.tree, dirname + '/' + fname)


def load_dt_policy(dirname, fname):
    f = open(dirname + '/' + fname, 'rb')
    dt_policy = pk.load(f)
    f.close()
    return dt_policy


class DTPolicy:
    def __init__(self, max_depth):
        self.max_depth = max_depth

        #####################
        ### START ADDITION
        #####################
        # initialize simulation
        self.config = import_test_configuration(config_file=PTH_TO_SETTINGS)
        self.sumo_cmd = set_sumo(self.config['gui'], self.config['sumocfg_file_name'], self.config['max_steps'])
        self.sumo_cmd[2] = PTH + self.sumo_cmd[2]

        self.TrafficGen = TrafficGenerator(
            self.config['max_steps'],
            self.config['n_cars_generated']
        )

        self.mod = None

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

    def fit(self, obss, acts):
        self.tree = DecisionTreeClassifier(max_depth=self.max_depth)
        self.tree.fit(obss, acts)

    def train(self, obss, acts, train_frac):
        obss_train, acts_train, obss_test, acts_test = split_train_test(obss, acts, train_frac)
        self.fit(obss_train, acts_train)
        log('Train accuracy: {}'.format(accuracy(self, obss_train, acts_train)), INFO)
        log('Test accuracy: {}'.format(accuracy(self, obss_test, acts_test)), INFO)
        log('Number of nodes: {}'.format(self.tree.tree_.node_count), INFO)

    def transform(self, state):
        return state

    def predict(self, obss):
        obss = self.transform(obss)
        return self.tree.predict(obss)

    def clone(self):
        clone = DTPolicy(self.max_depth)
        clone.tree = self.tree
        return clone

    def reset(self):
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

    def get_rollout_student(self, ep):
        """
        Get rollout with random seed.
        param ep: seed.
        """
        print("SEED = " + str(ep))
        #####################
        ### START ADDITION
        #####################
        # Here we took the original get_rollout_student function, and tailored it to work with
        # the Traffic-Control simulation
        self.reset()

        self.sim._TrafficGen.generate_routefile(seed=ep)
        traci.start(self.sim._sumo_cmd)

        obs, done = np.array(self.sim._get_state()), False
        rollout = []

        max_steps = self.sim._max_steps
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

            if steps != 0:
                rollout.append((old_state, old_action, reward))
                self.sim.handle_state_action_pair(old_state, old_action)

            # choose the light phase to activate, based on the current state of the intersection
            action = self.predict(np.array([current_state]))[0]

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

    def get_rollouts(self, n_batch_rollouts, j):
        """
        Get multiple rollouts for multiple seeds.
        param: n_batch_rollouts - amount of rollouts
        param: j - the multiple of 10 in the seeds
        """
        rollouts = []
        for i in range(n_batch_rollouts):
            print(i)
            rollouts.extend(self.get_rollout_student(10*j + i))
        return rollouts
