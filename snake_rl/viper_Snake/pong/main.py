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

from agent_1 import DQN
from viper_Snake.core.dt import *
from viper_Snake.pong.dqn import DQNPolicy
from viper_Snake.util.log import *
from viper_Snake.core.rl import train_dagger, test_policy

# Model number we want to run with
MODEL_NUM = 400

def learn_dt():
    """
    Main of VIPER.
    """
    # Parameters
    log_fname = '../snake_dt.log'
    model_num = MODEL_NUM
    max_depth = 12
    n_batch_rollouts = 10
    max_samples = 200000
    max_iters = 20#80
    train_frac = 0.8
    is_reweight = True
    n_test_rollouts = 10#50
    save_dirname = '../snake_pre/snake_reduced/snake'
    save_fname = 'dt_policy.pk'
    save_viz_fname = 'dt_policy.dot'
    is_train = False

    # Logging
    set_file(log_fname)

    env = Snake()

    # Data structures
    teacher = DQNPolicy(env, model_num)
    student = DTPolicy(max_depth, env)
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
    print(student.tree.tree_.node_count)

    # def feat_name(lane):
    #     return [lane + "[" + str(i) + "]" for i in range(10)]
    #

    #####################
    ### START ADDITION
    #####################
    # export tree

    feat_names = \
        ['APL_UP'] + ['APL_RIGHT'] + ['APL_DOWN'] + ['APL_LEFT'] + ['OBS_UP'] + ['OBS_RIGHT'] + ['OBS_DOWN'] + ['OBS_LEFT'] + ['DIR_UP'] + ['DIR_RIGHT'] + ['DIR_DOWN'] + ['DIR_LEFT']# + ['ACT_UP'] + ['ACT_RIGHT'] + ['ACT_DOWN'] + ['ACT_LEFT']

    print(feat_names)



    # DOT data
    dot_data = tree.export_graphviz(student.tree, out_file=None,class_names=["UP", "RIGHT", "LEFT", "DOWN"], feature_names=feat_names
                                    ,filled=True)

    # Draw graph
    graph = graphviz.Source(dot_data, format="svg")
    graph.render("final_student")
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


def print_size():
    # Parameters
    dirname = 'results/run9'
    fname = 'dt_policy.pk'

    # Load decision tree
    dt = load_dt_policy(dirname, fname)

    # Size
    print(dt.tree.tree_.node_count)

def cnt_undesirable(N):
    save_dirname = os.path.abspath('../snake_pre/snake_reduced/snake')
    save_fname = 'dt_policy_reduced.pk'
    student = load_dt_policy(save_dirname, save_fname)
    student.post_save()

    rats = []
    rews = []

    try:
        with open("../snake_pre/stud_ratios.pkl", 'rb') as f:
            rats = pickle.load(f)

        with open("../snake_pre/stud_rewards.pkl", 'rb') as f:
            rews = pickle.load(f)
    except:
        x=1

    if len(rats) != len(rews):
        print("ERROR LENGTHGGG")
        exit(1)

    for i in range(len(rats), N):
        rollout, rat = student.get_rollout_student(i, with_meta=True)
        #rollouts.extend(student.get_rollout_student(10 + i))


        rew = sum((rew for _, _, rew in rollout))

        rats.append(rat)
        rews.append(rew)

        print("Iteration " + str(i) + ":")
        print(rats)
        print(rews)

        with open("../snake_pre/stud_ratios.pkl", 'wb+') as f:
            pickle.dump(rats, f)

        with open("../snake_pre/stud_rewards.pkl", 'wb+') as f:
            pickle.dump(rews, f)

    print(np.average(rats))
    print(np.average(rews))

    return rats, rews


if __name__ == '__main__':
    learn_dt()
    #cnt_undesirable(100)