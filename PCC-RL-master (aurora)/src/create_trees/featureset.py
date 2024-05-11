import sys
from enum import Enum

import numpy as np
import tensorflow as tf
import os


if int(sys.argv[1]) == 0:
    from gym.eval_aurora import get_model_output, get_input

    try:
        run_number = int(sys.argv[3]) + int(sys.argv[4])
    except Exception as e:
        run_number = 1602

    working_model = 1602

    PTH_TO_GYM = '../gym/'
    MOD_PTH = PTH_TO_GYM + 'runs\\runs' + str(run_number) + '\\model'

    if not os.path.exists(MOD_PTH):
        MOD_PTH = PTH_TO_GYM + 'runs/runs' + str(working_model) + '/model'
        print("WORKING ON MODEL")
        print(working_model)

    SESS = tf.Session(graph=tf.Graph())
    tf.saved_model.loader.load(SESS, [tf.saved_model.SERVING], MOD_PTH)

else:
    from eval_aurora import get_model_output, get_input

    try:
        run_number = int(sys.argv[3]) + int(sys.argv[4])
    except Exception as e:
        run_number = 1602

    working_model = 1602

    PTH_TO_GYM = '../gym/'

    MOD_PTH = PTH_TO_GYM + 'runs/runs' + str(run_number) + '/model'
    if not os.path.exists(MOD_PTH):
        MOD_PTH = PTH_TO_GYM + 'runs/runs' + str(working_model) + '/model'
        print("WORKING ON MODEL")
        print(working_model)


    SESS = tf.Session(graph=tf.Graph())
    tf.saved_model.loader.load(SESS, [tf.saved_model.SERVING], MOD_PTH)
"""
The features that our decision tree learns on - i.e. the grammar.
These features are specifically for Aurora:
    1. INDEX - the value of some index
    2. DIFF - The difference between the values in two indices
    3. SIGN - The sign of the difference between values in two indices
    4. AVERAGE - The average of some indices

"""

"""
HOW TO ADD A FEATURE:
1. Add the name to TYPE_FEAT
2. Add an 'elif' to __init__
2. Add an 'elif' clause to calc_feat
3. Add an 'elif' clause to parse_line
"""


# TYPES OF FEATURES
class TYPE_FEAT(Enum):
    INDEX = 1
    DIFF = 2
    SIGN = 3
    AVERAGE = 4
    RES = 5


class Feature:
    # constants
    INDEX1 = 0
    INDEX2 = 1

    def __init__(self, type, indices):
        """
        Initialize with indices
        """
        self.type = type
        self.indices = indices

        # if INDEX, then return the value
        if self.type == TYPE_FEAT.INDEX:
            self.name = 'IDX-' + str(self.indices[Feature.INDEX1])
        # if DIFF, then return difference between the two
        elif self.type == TYPE_FEAT.DIFF:
            self.name = 'DFF-' + str(self.indices[Feature.INDEX1]) + '~' + str(self.indices[Feature.INDEX2])
        # if DIFF, then return the sign of the difference between the two
        elif self.type == TYPE_FEAT.SIGN:
            self.name = 'SGN-' + str(self.indices[Feature.INDEX1]) + '~' + str(self.indices[Feature.INDEX2])
        # if AVERAGE, then return the average of the indices
        elif self.type == TYPE_FEAT.AVERAGE:
            self.name = 'AVG-' + '~'.join([str(ind) for ind in self.indices])
        # if RES, then return the res
        elif self.type == TYPE_FEAT.RES:
            self.name = 'RES'

    def calc_feat(self, arr, res=(False,0)):
        """
        Extract the features from a state
        """
        # if INDEX, then return the value
        if self.type == TYPE_FEAT.INDEX:
            return arr[self.indices[Feature.INDEX1]]
        # if DIFF, then return difference between the two
        elif self.type == TYPE_FEAT.DIFF:
            return arr[self.indices[Feature.INDEX1]] - arr[self.indices[Feature.INDEX2]]
        # if DIFF, then return the sign of the difference between the two
        elif self.type == TYPE_FEAT.SIGN:
            return np.sign(arr[self.indices[Feature.INDEX1]] - arr[self.indices[Feature.INDEX2]])
        # if AVERAGE, then return the average of the indices
        elif self.type == TYPE_FEAT.AVERAGE:
            return np.average((np.asarray(arr))[self.indices])
        # if RES, then return the result
        elif self.type == TYPE_FEAT.RES:
            # action we took
            if res[0] == False:
                return get_model_output(SESS=SESS, obs=get_input(arr))
                print("GOT OUTPUT FROM EXISTING MODEL")
            else:
                return res[1]


class Featureset:
    """
    Initialize default featureset
    """
    def __init__(self):
        self.feats = []

    def parse_subline(self, type, line):
        """
        Parse a line in the format i1~i2~i3~...~iN_i1~i2~i3~...~iM_..._i1~i2~i3~...~iK
        """
        # split to different features
        features = line.split('_')
        for feature in features:
            # get indices
            indices = [int(ind) for ind in feature.split('~')]
            # create new feature
            self.feats.append(Feature(type, indices))

    def parse_line(self, line):
        """
        Parse one line in the featureset file
        """
        line = line.strip()
        if line.startswith('INDEX'):
            feat_type = TYPE_FEAT.INDEX
        elif line.startswith('DIFF'):
            feat_type = TYPE_FEAT.DIFF
        elif line.startswith('SIGN'):
            feat_type = TYPE_FEAT.SIGN
        elif line.startswith('AVERAGE'):
            feat_type = TYPE_FEAT.AVERAGE
        elif line.startswith('RES'):
            feat_type = TYPE_FEAT.RES
            self.feats.append(Feature(feat_type, []))
            return
        subline = line.split('-')[1]
        # cut text
        self.parse_subline(feat_type, subline)

    def parse_file(self, file):
        """
        Parse a featureset file, line by line
        """
        f = open(file, 'r')
        for line in f:
            self.parse_line(line)
        return self

    def expand_arr(self, arr, res=(False, 0)):
        """
        Extract all features from a state.
        param: arr - the state we want to extract features from.
        """
        new_arr = []
        for feat in self.feats:
            new_arr.append(feat.calc_feat(arr, res))
        return new_arr
