import os
from enum import Enum

import numpy as np
import sys
import tensorflow as tf

# on cluster or not

if int(sys.argv[1]) == 0:
    from src.gym.eval_aurora import get_model_output, get_input

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
    import create_trees.eval_network_restart_from_clean_v1 as eval_network
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

class TYPE(Enum):
    """
    The type of rule
    """
    DIFF = 1
    BOUND = 2
    COMPARE = 3
    RES = 4


class Rule:
    """
    A generic rule
    """

    INDEX1 = 0
    INDEX2 = 1

    def __init__(self):
        """
        Initialize rules.
        """
        self.indices = []

        # self.index1 = -1
        # self.index2 = -1

        self.comp = 0
        self.value = -1
        self.type = -1


    def parse_index(self, ind):
        """
        Parse the index.
        """
        return int(ind[1])

    def parse_comp(self, comp):
        """
        Parse the operation.
        """
        if comp == '>':
            return 1
        if comp == '<':
            return -1
        if comp == '=':
            return 0


    def parse_value(self, val):
        """
        Parse the value.
        """
        return float(val)

    def parse_line(self, line):
        """
        Parse line.
        Options for line:
            ii > ij
            ii > 0
            ii - ij < 1
            R > 0 / R < 0 / R = 0
        """
        parts = line.split(' ')
        # RER
        if parts[0] == 'R':
            self.type = TYPE.RES
            self.comp = self.parse_comp(parts[1])
            self.value = self.parse_value(parts[2])
        # DIFF
        elif len(parts) > 3:
            self.indices = [self.parse_index(parts[0]), self.parse_index(parts[2])]
            # self.index1 = self.parse_index(parts[0])
            # self.index2 = self.parse_index(parts[2])
            self.comp = self.parse_comp(parts[3])
            self.value = self.parse_value(parts[4])
            self.type = TYPE.DIFF
        else:
            # BOUND
            if parts[2].startswith('i'):
                self.indices = [self.parse_index(parts[0]), self.parse_index(parts[2])]
                # self.index1 = self.parse_index(parts[0])
                # self.index2 = self.parse_index(parts[2])
                self.comp = self.parse_comp(parts[1])
                self.type = TYPE.BOUND
            # COMPARE
            else:
                self.indices.append(self.parse_index(parts[0]))
                # self.index1 = self.parse_index(parts[0])
                self.comp = self.parse_comp(parts[1])
                self.value = self.parse_value(parts[2])
                self.type = TYPE.COMPARE
        return self


    def check_arr(self, arr):
        """
        Check if state is undesirable per the specific rule
        param: arr - the state
        """
        # DIFF
        if self.type == TYPE.DIFF:
            tempval = (arr[self.indices[Rule.INDEX1]] - arr[self.indices[Rule.INDEX2]] - self.value)
            # if comp is '=' and tempval is 0, then they are equal
            if self.comp == 0 and tempval == 0:
                return True
        # COMPARE
        if self.type == TYPE.COMPARE:
            tempval = (arr[self.indices[Rule.INDEX1]] - self.value)
            if self.comp == 0 and tempval == 0:
                return True
        # BOUND
        if self.type == TYPE.BOUND:
            tempval = (arr[self.indices[Rule.INDEX1]] - arr[self.indices[Rule.INDEX2]])
            if self.comp == 0 and tempval == 0:
                return True
        # RES
        # if self.type == TYPE.RES:
        #     # if we want to ignore the 'res'
        #     if no_res:
        #         return True
        #     tempval = get_model_output(SESS=SESS, obs=get_input(arr))
        #     if self.comp == 0 and tempval == 0:
        #         return True
        # # if the tempval abides by the comp '<' or '>'
        return np.sign(tempval*self.comp) == 1


    def check_only_action(self, arr, action):
        """
        Check if action is undesirable for that state.
        param: arr - state
        param: action - action
        """
        if action == False:
            action = get_model_output(SESS=SESS, obs=get_input(arr))
            print("GOT OUTPUT FROM EXISTING MODEL")
        if self.comp == 0 and action == 0:
            return True
        return np.sign(action*self.comp) == 1

    def check(self, arr, action):
        """
        Check if state-action pair is undesirable.
        """
        if self.type == TYPE.RES:
            return self.check_only_action(arr, action)
        return self.check_arr(arr)



class Ruleset:
    """
    A set of rules.
    """
    def __init__(self):
        self.rules = []

    def parse_file(self, file):
        """
        Parse rules file.
        """
        f = open(file, 'r')
        for line in f:
            if line.startswith('#'):
                continue
            self.rules.append(Rule().parse_line(line.strip()))
        f.close()
        return self

    def check(self, arr, action=False):
        """
        Check for all of the rules, if the state-action pair abides by them.
        This means we check if our state-action is desirable or undesirable.
        param: arr - the state
        param: action - the action
        """
        bool = True
        for rule in self.rules:
            if not bool:
                break
            bool = rule.check(arr, action)
        return bool

    def get_res_rule(self):
        for rule in self.rules:
            if rule.type == TYPE.RES:
                return rule.comp