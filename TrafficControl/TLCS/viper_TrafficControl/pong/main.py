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
import pickle

import graphviz
import numpy as np
from sklearn import tree

from TrafficControl.TLCS.viper_TrafficControl.core.dt import *
from TrafficControl.TLCS.viper_TrafficControl.pong.dqn import DQNPolicy
from TrafficControl.TLCS.viper_TrafficControl.util.log import *
from TrafficControl.TLCS.viper_TrafficControl.core.rl import train_dagger, test_policy

import os

MODEL_NUM = 400

def learn_dt():
    # Parameters
    log_fname = '../traffic_dt.log'
    model_num = MODEL_NUM
    max_depth = 12
    n_batch_rollouts = 10
    max_samples = 200000
    max_iters = 20#80
    train_frac = 0.8
    is_reweight = True
    n_test_rollouts = 10#50
    save_dirname = '../traffic_control/traffic'
    save_fname = 'dt_policy.pk'
    save_viz_fname = 'dt_policy.dot'
    is_train = False

    # Logging
    set_file(log_fname)

    # Data structures
    teacher = DQNPolicy(model_num)
    student = DTPolicy(max_depth)
    # Train student
    if is_train:
        student = train_dagger(teacher, student, max_iters, n_batch_rollouts, max_samples,
                               train_frac, is_reweight, n_test_rollouts)
        save_dt_policy(student, save_dirname, save_fname)
        save_dt_policy_viz(student, save_dirname, save_viz_fname)
    else:
        student = load_dt_policy(save_dirname, save_fname)

    # Test student
    # rew = test_policy(student, n_test_rollouts, 100)
    # log('Final reward: {}'.format(rew), INFO)
    log('Number of nodes: {}'.format(student.tree.tree_.node_count), INFO)

    #####################
    ### START ADDITION
    #####################
    # visualize the tree
    def feat_name(lane):
        return [lane + "[" + str(i) + "]" for i in range(10)]

    feat_names = feat_name("W2E") + feat_name("W2TL")\
                 + feat_name("E2W") + feat_name("E2TL")\
                 + feat_name("N2S") + feat_name("N2TL")\
                 + feat_name("S2N") + feat_name("S2TL")

    print(feat_names)

    # DOT data
    dot_data = tree.export_graphviz(student.tree, out_file=None,class_names=["NS", "NSL", "EW", "EWL"], feature_names=feat_names
                                    ,filled=True)

    # Draw graph
    graph = graphviz.Source(dot_data, format="svg")
    graph.render("final_student_new")
    #####################
    ### END ADDITION
    #####################


# def bin_acts():
#     # Parameters
#     seq_len = 10
#     n_rollouts = 10
#     log_fname = 'pong_options.log'
#     model_path = 'model-atari-pong-1/saved'
#
#     # Logging
#     set_file(log_fname)
#
#     # Data structures
#     env = get_pong_env()
#     teacher = DQNPolicy(env, model_path)
#
#     # Action sequences
#     seqs = get_action_sequences(env, teacher, seq_len, n_rollouts)
#
#     for seq, count in seqs:
#         log('{}: {}'.format(seq, count), INFO)

#
# def print_size():
#     # Parameters
#     dirname = 'results/run9'
#     fname = 'dt_policy.pk'
#
#     # Load decision tree
#     dt = load_dt_policy(dirname, fname)
#
#     # Size
#     print(dt.tree.tree_.node_count)

def cnt_undesirable(N):
    """
    Get VIPER performance - undesirable behavior ratio and average reward arrays.
    param: N - number of rollouts to average
    """
    save_dirname = r'viper_TrafficControl\traffic_control\traffic'
    print(os.curdir)
    save_fname = 'dt_policy.pk'
    # get student
    student = load_dt_policy(save_dirname, save_fname)


    prms = []
    prms_rews = []

    # load saved data
    try:
        with open("stud_ratios.pkl", 'rb') as f:
            prms = pickle.load(f)

        with open("stud_rewards.pkl", 'rb') as f:
            prms_rews = pickle.load(f)
    except:
        x=1

    if len(prms) != len(prms_rews):
        print("ERROR LENGTHGGG")
        exit(1)

    #
    for i in range(len(prms), N):
        rollout = student.get_rollout_student(i)
        #rollouts.extend(student.get_rollout_student(10 + i))
        # get undesirable behavior ratios
        prm = student.sim.params["bad_state_cntr"] / student.sim.params["all_state_cntr"]
        prms.append(prm)
        print(prm)

        # get rewards (only negative ones, because this is how we measure ourselves without VIPER)
        neg_rewards = sum((rew for _, _, rew in rollout if rew < 0))

        prms_rews.append(neg_rewards)
        print(neg_rewards)

        # pickle arrays
        with open("stud_ratios.pkl", 'wb+') as f:
            pickle.dump(prms, f)

        with open("stud_rewards.pkl", 'wb+') as f:
            pickle.dump(prms_rews, f)

    # get average undesirable behavior ratios, and average rewards
    print(np.average(prms))
    print(np.average(prms_rews))

    # get performance
    return prms, prms_rews


if __name__ == '__main__':
    learn_dt()
    #cnt_undesirable(30)