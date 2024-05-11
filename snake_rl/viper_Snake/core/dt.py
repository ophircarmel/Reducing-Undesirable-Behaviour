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
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz

import snake_env
from Tree.utils_for_tree import orig_state_action_pair_to_state_dict, from_state_dict_to_learnable_state, is_bad_prop
from snake_env import Snake
from ..util.log import *


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

    dt_policy.pre_save()
    f = open(dirname + '/' + fname, 'wb')
    pk.dump(dt_policy, f)
    f.close()

    dt_policy.post_save()

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
    def __init__(self, max_depth, env):
        self.max_depth = max_depth
        self.env = env

    def fit(self, obss, acts):
        self.tree = DecisionTreeClassifier(max_depth=self.max_depth)
        self.tree.fit(obss, acts)

    def train(self, obss, acts, train_frac):
        obss = np.array(obss)
        acts = np.array(acts)
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
        clone = DTPolicy(self.max_depth, Snake())
        clone.tree = self.tree
        return clone

    def reset(self):
        return self.env.reset()

    def get_rollout_student(self, ep, max_steps=1000, with_meta=False):
        undesirable_cntr = 0
        desirable_cntr = 0
        steps = 0
        score = 0
        rollout = []

        state = np.array(self.reset())
        state = np.reshape(state, (1, self.env.state_space))

        action = self.predict([np.array(state[0])])[0]

        next_state, reward, done, _ = self.env.step(action)
        next_state = np.reshape(next_state, (1, self.env.state_space))

        state = next_state
        action = self.predict([np.array(state[0])])[0]

        while not done:

            next_state, reward, done_snake, _ = self.env.step(action)
            next_state = np.reshape(next_state, (1, self.env.state_space))
            score += reward

            rollout.append((state, action, reward))

            state = next_state
            action = self.predict([np.array(state[0])])[0]

            #####################
            ### START ADDITION
            #####################
            # count amount of desirable and undesirable actions

            # state_dict
            d = orig_state_action_pair_to_state_dict(state[0].tolist(), action)
            # get label
            label = is_bad_prop(d, prop_code=1)
            # add to relevant counter
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

        if with_meta:
            return rollout, undesirable_cntr / (desirable_cntr + undesirable_cntr)
        else:
            return rollout

    def get_rollouts(self, n_batch_rollouts, j):
        rollouts = []
        scores = []
        und_cntrs = []
        d_cntrs = []
        for i in range(n_batch_rollouts):
            print(i)
            rollout = self.get_rollout_student(i)
            # scores.extend(score)
            # und_cntrs.extend(und_cntr)
            # d_cntrs.extend(d_cntr)
            rollouts.extend(rollout)
        return rollouts

    def pre_save(self):
        self.env = None

    def post_save(self):
        self.env = Snake()
