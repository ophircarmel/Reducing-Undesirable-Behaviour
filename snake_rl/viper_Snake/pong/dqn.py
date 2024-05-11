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
import pickle
import sys

import numpy as np
import tensorflow as tf


# As described in https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
from Tree.utils_for_tree import orig_state_action_pair_to_state_dict, is_bad_prop
from agent_1 import DQN


class DQNPolicy:
    def __init__(self, env, mod_num):
        #####################
        ### START ADDITION
        #####################
        # add our model
        mod_pth = r"..\..\models\model_" + str(mod_num) + "\\"
        with open(mod_pth + "episode-49-agent.pkl", 'rb') as f:
            self.mod = pickle.load(f)
        #####################
        ### END ADDITION
        #####################
        self.env = env

    def reset(self):
        return self.env.reset()

    def predict_q_one(self, state):
        state = np.reshape(state, (1, self.env.state_space))
        act_values = self.mod.model.predict(state)[0]
        return act_values

    def predict_q(self, states):
        qs = []
        for state in states:
            qs.append(self.predict_q_one(state))

        return qs

    def predict(self, states):
        acts = []
        for state in states:
            acts.append(self.mod.act(state))

        return acts

    def get_rollout_teacher(self, ep, max_steps=1000):
        undesirable_cntr = 0
        desirable_cntr = 0
        steps = 0
        score = 0
        rollout = []

        state = np.array(self.reset())
        state = np.reshape(state, (1, self.env.state_space))

        action = self.predict(np.array([state]))[0]

        next_state, reward, done, _ = self.env.step(action)
        next_state = np.reshape(next_state, (1, self.env.state_space))

        state = next_state
        action = self.predict(np.array([state]))[0]

        while not done:

            next_state, reward, done_snake, _ = self.env.step(action)
            next_state = np.reshape(next_state, (1, self.env.state_space))
            score += reward

            rollout.append((state, action, reward))

            state = next_state
            action = self.predict(np.array([state]))[0]

            #####################
            ### START ADDITION
            #####################
            # count desirable and undesirable state-action pairs

            d = orig_state_action_pair_to_state_dict(state[0].tolist(), action)
            label = is_bad_prop(d, prop_code=1)
            if d['DIR'] != [0, 0, 0, 0]:
                if label:
                    undesirable_cntr += 1
                else:
                    desirable_cntr += 1
            #####################
            ### END ADDITION
            #####################

            steps += 1

            done = done_snake or (steps >= max_steps)

        #return rollout, score, undesirable_cntr, desirable_cntr
        return rollout

    def get_rollouts(self, n_batch_rollouts):
        rollouts = []
        for i in range(n_batch_rollouts):
            print(i)
            rollouts.extend(self.get_rollout_teacher(i))
        return rollouts

