from enum import Enum

import numpy as np

"""
The features that our decision tree tries to learn on.
These features are specifically for Snake:
    1. INDEX - the value of some index
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


class Feature:
    # constants
    INDEX1 = 0

    """
    Initialize with indices
    """

    def __init__(self, type, indices):
        self.type = type
        self.indices = indices

        # if INDEX, then return the value
        if self.type == TYPE_FEAT.INDEX:
            self.name = 'IDX-' + str(self.indices[Feature.INDEX1])

    """
    Calculate the feature from an array
    """

    def calc_feat(self, arr):
        # if INDEX, then return the value
        if self.type == TYPE_FEAT.INDEX:
            return arr[self.indices[Feature.INDEX1]]


class Featureset:
    """
    Initialize default featureset
    """
    def __init__(self):
        self.feats = []

    """
    Parse a line in the format i1~i2~i3~...~iN_i1~i2~i3~...~iM_..._i1~i2~i3~...~iK
    """
    def parse_subline(self, type, line):
        # split to different features
        features = line.split('_')
        for feature in features:
            # get indices
            indices = [int(ind) for ind in feature.split('~')]
            # create new feature
            self.feats.append(Feature(type, indices))

    """
    Parse one line in the featureset file
    """
    def parse_line(self, line):
        line = line.strip()
        if line.startswith('INDEX'):
            feat_type = TYPE_FEAT.INDEX
        subline = line.split('-')[1]
        # cut text
        self.parse_subline(feat_type, subline)

    """
    Parse a featureset file, line by line
    """
    def parse_file(self, file):
        f = open(file, 'r')
        for line in f:
            self.parse_line(line)
        return self

    """
    Calculate all features and return an array of all calulated features of an array
    """
    def expand_arr(self, arr):
        new_arr = []
        for feat in self.feats:
            new_arr.append(feat.calc_feat(arr))
        return new_arr
