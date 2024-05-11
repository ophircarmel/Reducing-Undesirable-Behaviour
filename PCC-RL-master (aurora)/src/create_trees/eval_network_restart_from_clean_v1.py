# working 30-8-2021
import random
import sys

import numpy as np
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import graph_util
import tensorflow as tf
import os

GRAPH_PB_PATH = os.path.abspath("new_model")
if not os.path.exists(GRAPH_PB_PATH):
    GRAPH_PB_PATH = os.path.abspath("new_model")


LG = random.uniform(-0.001, 0.001)
LR = random.uniform(1, 1)

def init_srs():
    l = [0 for _ in range(30)]
    for i in range(10):
        l[3 * i + 0] = LG
        l[3 * i + 1] = LR
        l[3 * i + 2] = 0
    return l


INIT_SRS = init_srs()

def get_input(srss):
    for i in range(10):
        INIT_SRS[3*i+2] = srss[i]
    return INIT_SRS


SESS = tf.Session(graph=tf.Graph())
tf.saved_model.loader.load(SESS, [tf.saved_model.SERVING], GRAPH_PB_PATH)


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

def get_model_output(srss):
    inputValues = [np.array(get_input(srss))]
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
    out = SESS.run(outputName + ":0", feed_dict=feed_dict)  # todo get output name!
    return out[0][0]
