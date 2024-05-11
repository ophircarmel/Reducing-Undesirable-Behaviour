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

import sys
from random import random

import gym
import numpy as np
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

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.policies import FeedForwardPolicy
from stable_baselines import PPO1
import os
import inspect

if len(sys.argv) > 4:
    run_number = int(sys.argv[3]) + int(sys.argv[4])
else:
    run_number = 1602
on_cluster = (int(sys.argv[1]) == 1)

PTH_TO_GYM = '' ############## enter path to gym folder
MOD_PTH = PTH_TO_GYM + 'runs\\runs' + str(run_number) + '\\model'

def init_srs():
    l = [0 for _ in range(30)]
    for i in range(10):
        l[3 * i + 0] = 0
        l[3 * i + 1] = 1
        l[3 * i + 2] = 0
    return l

def get_input(srss):
    fin_srss = init_srs()
    for i in range(10):
        fin_srss[3*i+2] = srss[i]
    return fin_srss

def genShapeMap(ops):
    """
    Function to set input operations
    Arguments:
        [ops]: (tf.op) list representing input
    """
    shapeMap = dict()
    for op in ops:
        try:
            shape = tuple(op.outputs[0].shape.as_list())
            shapeMap[op] = shape
        except:
            shapeMap[op] = [None]
    return shapeMap

def get_model_output(SESS, obs):
    """
    This function evaluates the model on the current state.
    param: SESS - the session
    param: obs - current state.
    """
    inputValues = [np.array(obs)]
    inputNames = np.array(["input/Ob"])
    outputName = "model/pi/add"  # "output/strided_slice_1"

    inputOps = []
    for i in inputNames:
        inputOps.append(SESS.graph.get_operation_by_name(i))
    shapeMap = genShapeMap(inputOps)
    inputValuesReshaped = []
    for j in range(len(inputOps)):
        inputOp = inputOps[j]
        inputShape = shapeMap[inputOp]
        inputShape = [i if i is not None else 1 for i in inputShape]
        # Try to reshape given input to correct shape
        inputValuesReshaped.append(inputValues[j].reshape(inputShape))
    inputNames = [o.name + ":0" for o in inputOps]
    feed_dict = dict(zip(inputNames, inputValuesReshaped))
    out = SESS.run(outputName + ":0", feed_dict=feed_dict)
    return out[0][0]


def test_loop():
    """
    Test a model.
    This tests the model with the run_number entered as an argument.
    When doing env.step(), it does the logic inside of network_sim, and logs the metrics using wandb.
    """
    ITERS = 100

    env = gym.make('PccNs-v0')

    SESS = tf.Session(graph=tf.Graph())
    tf.saved_model.loader.load(SESS, [tf.saved_model.SERVING], MOD_PTH)

    iters = ITERS

    for iter in range(iters):
        print("\n\n********** ITERATION " + str(iter) + " OUT OF " + str(ITERS) + " **********")
        sub_iters = 20
        for i in range(sub_iters):

            observation = env.reset()
            action = get_model_output(SESS, observation)
            done = False

            while not done:
                observation, reward, done, info = env.step([action])


if __name__ == '__main__':
    test_loop()