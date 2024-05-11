# Copyright 2019 Nathan Jay and Noga Rotman
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
import datetime
import sys
import gym
import network_sim
import tensorflow as tf

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.policies import FeedForwardPolicy
from stable_baselines import PPO1
import os
import inspect



currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
sys.path.insert(0, parentdir + '/common')
print(sys.path)
from common.simple_arg_parse import arg_or_default

#################
# START ADDITION
#################
"""
ARGUMENTS:
sys.argv[1] - 1 if it is run on cluster, 0 if on my cpu
sys.argv[2] - 1 if it is a test run, 0 if train
sys.argv[3] - run number
sys.argv[4] - place in array
sys.argv[5] - ruleset name
sys.argv[6] - featureset name
sys.argv[7] - treefile name
sys.argv[8] - message
sys.argv[9] - alpha downscaling 
sys.argv[10] - training_steps
sys.argv[11] - time_steps
"""

# The number of training steps
# DEFAULT_TRAINING_STEPS = 6
DEFAULT_TRAINING_STEPS = 10
# The number of iterations out of total timesteps
DEFAULT_ITERS_TIMESTEPS = 80

on_cluster = int(sys.argv[1])
train = (int(sys.argv[2]) == 0)  # 0 for training, 1 for testing
run_number = int(sys.argv[3]) + int(sys.argv[4])
ruleset_name = sys.argv[5]
featureset_name = sys.argv[6]
treefile_name = sys.argv[7]
message = sys.argv[8]
alpha_downscaling = float(sys.argv[9])
if len(sys.argv) > 10:
    training_steps = int(sys.argv[10])
    time_steps = int(sys.argv[11])
else:
    training_steps = DEFAULT_TRAINING_STEPS
    time_steps = DEFAULT_ITERS_TIMESTEPS

import wandb
import pickle

if train:
    config = {
        "run_number": run_number,
        "alpha_downscaling": alpha_downscaling,
        "training_steps": training_steps,
        "time_steps": time_steps
    }

    wandb.init(
        project="PCC-RL-GRAPHING",
        entity="", # name of entity. One can just comment out the wandb parts to not need to open a user
        name='Run ' + str(run_number) + ' - Ruleset = ' + ruleset_name + ' - DOWNSCALING = ' + str(alpha_downscaling) + " - " + message,
        save_code=True, tags = [str(run_number)[:2]], id='id' + str(run_number), config=config)

else:
    # test
    config = {
        "run_number": run_number,
        "alpha_downscaling": alpha_downscaling
    }

    wandb.init(
        project="PCC-RL-GRAPHING",
        entity="", # name of entity. One can just comment out the wandb parts to not need to open a user
        name='Test of Run ' + str(run_number) + ' - DOWNSCALING = ' + str(alpha_downscaling),
        save_code=True,
        tags=[str(run_number)[:2]],
        id='tst' + str(run_number),
        config=config
    )

#################
# END ADDITION
#################

arch_str = arg_or_default("--arch", default="32,16")
if arch_str == "":
    arch = []
else:
    arch = [int(layer_width) for layer_width in arch_str.split(",")]
print("Architecture is: %s" % str(arch))

training_sess = None


class MyMlpPolicy(FeedForwardPolicy):

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **_kwargs):
        super(MyMlpPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse,
                                          net_arch=[{"pi": arch, "vf": arch}],
                                          feature_extraction="mlp", **_kwargs)
        global training_sess
        training_sess = sess


env = gym.make('PccNs-v0')
# env = gym.make('CartPole-v0')

# train or test
if not train:
    if on_cluster:
        from eval_aurora import test_loop
    else:
        from src.gym.eval_aurora import test_loop
    # test the model
    test_loop()
    exit(0)


gamma = arg_or_default("--gamma", default=0.99)
print("gamma = %f" % gamma)
model = PPO1(MyMlpPolicy, env, verbose=1, schedule='constant', timesteps_per_actorbatch=8192, optim_batchsize=2048,
             gamma=gamma)

#################
# START ADDITION
#################

if on_cluster:
    PTH_TO_GYM = '' # enter the path to the gym folder
else:
    PTH_TO_GYM = '' # enter the path to the gym folder

# create dir
default_export_dir = PTH_TO_GYM + "/runs/runs" + str(run_number) + "/model/"
# If dir exists
if os.path.isdir(default_export_dir):
    import shutil

    shutil.rmtree(default_export_dir)
# create pth for logs
logpth = PTH_TO_GYM + '/runs/runs' + str(run_number) + "/logs"
if not os.path.isdir(logpth):
    os.mkdir(logpth)

measure_times = False
# if we want to measure times
if measure_times:
    datetimes = []

    for i in range(5):
        timestamp_start = datetime.datetime.now()

        model.learn(total_timesteps=(20*time_steps*410))

        timestamp_end = datetime.datetime.now()

        datetimes.append(timestamp_end - timestamp_start)

    for da in datetimes:
        print(da)
    exit(0)

#################
# END ADDITION
#################

for i in range(0, training_steps):
    with model.graph.as_default():
        saver = tf.train.Saver()
        saver.save(training_sess, "./pcc_model_%d.ckpt" % i)
    # x ITERS = 410 * (20 * x)
    # 400 step() call between each reward print
    model.learn(total_timesteps=(20 * time_steps * 410))

##
#   Save the model to the location specified below.
##

export_dir = arg_or_default("--model-dir", default=default_export_dir)
with model.graph.as_default():
    pol = model.policy_pi  # act_model

    obs_ph = pol.obs_ph
    act = pol.deterministic_action
    sampled_act = pol.action

    obs_input = tf.saved_model.utils.build_tensor_info(obs_ph)
    outputs_tensor_info = tf.saved_model.utils.build_tensor_info(act)
    stochastic_act_tensor_info = tf.saved_model.utils.build_tensor_info(sampled_act)
    signature = tf.saved_model.signature_def_utils.build_signature_def(
        inputs={"ob": obs_input},
        outputs={"act": outputs_tensor_info, "stochastic_act": stochastic_act_tensor_info},
        method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)

    # """
    signature_map = {tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                         signature}

    model_builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
    model_builder.add_meta_graph_and_variables(model.sess,
                                               tags=[tf.saved_model.tag_constants.SERVING],
                                               signature_def_map=signature_map,
                                               clear_devices=True)
    model_builder.save(as_text=True)

timestamp_end = datetime.datetime.now()

# with open(export_dir + "/timestamp.txt", "w+") as f:
#     f.write(str(timestamp_end - timestamp_start) + "\n")