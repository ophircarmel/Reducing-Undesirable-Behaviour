from enum import Enum

import numpy as np

"""
The type of rule
"""
class TYPE(Enum):
    ON = 1

"""
A generic rule
"""
class Rule:

    INDEX1 = 0

    """
    Initialize rules.
    """
    def __init__(self):
        self.indices = []

        self.comp = 0
        self.value = -1
        self.type = -1

    """
    Parse the index.
    """
    def parse_index(self, ind):
        return int(ind[1])

    """
    Parse the operation.
    """
    def parse_comp(self, comp):
        if comp == '>':
            return 1
        if comp == '<':
            return -1
        if comp == '=':
            return 0

    """
    Parse the value.
    """
    def parse_value(self, val):
        return int(val)

    """
    Parse line.
    Options for line:
        ii > ij
        ii > 0
        ii - ij < 1
        R > 0 / R < 0 / R = 0
    """
    def parse_line(self, line):
        parts = line.split(' ')
        # COMPARE
        self.indices.append(self.parse_index(parts[0]))
        # self.index1 = self.parse_index(parts[0])
        self.comp = self.parse_comp(parts[1])
        self.value = self.parse_value(parts[2])
        self.type = TYPE.COMPARE
        return self

    """
    Check if array abides by the rule
    """
    def check(self, arr, no_res=False):
        # COMPARE
        if self.type == TYPE.COMPARE:
            tempval = (arr[self.indices[Rule.INDEX1]] - self.value)
            if self.comp == 0 and tempval == 0:
                return True
        return np.sign(tempval*self.comp) == 1


"""
A set of rules.
"""
class Ruleset:
    def __init__(self):
        self.rules = []

    def parse_file(self, file):
        f = open(file, 'r')
        for line in f:
            if line.startswith('#'):
                continue
            self.rules.append(Rule().parse_line(line.strip()))
        f.close()
        return self

    def check(self, arr, no_res=False):
        bool = True
        for rule in self.rules:
            if not bool:
                break
            bool = rule.check(arr, no_res)
        return bool

    def get_res_rule(self):
        for rule in self.rules:
            if rule.type == TYPE.RES:
                return rule.comp