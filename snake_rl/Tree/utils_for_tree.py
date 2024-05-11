"""
Utility functions for classifying and labeling.
Three type of states:
Original state
State_Dict
Learnable_State
"""
from Tree.features import Value, IsEqual
import Tree.traffic_features as traffic_feats

# consts
CATEGORY_NAMES = ['APL', 'OBS', 'DIR', 'ACT']
CATEGORY_DICT = {'APL': 0, 'OBS' : 1, 'DIR' : 2, 'ACT' : 3}
NUM_TO_DIR = {0 : 'U', 1 : 'R', 2 : 'D', 3 : 'L'}
DIR_TO_LIST_DICT = {'U' : [1, 0, 0, 0], 'R' : [0, 1, 0, 0], 'D' : [0, 0, 1, 0], 'L': [0, 0, 0, 1]}

# the dictionary of all our features - for snake grammar
FEATURE_DICT = {
    "Value" : [(i,j) for j in range(len(CATEGORY_NAMES)) for i in range(4)],
    "IsEqual" : [(i,j,k) for i in range(4) for j in range(len(CATEGORY_NAMES)) for k in range(j, len(CATEGORY_NAMES)) if j!=k]
}

def get_list_of_features_from_dict(state, feature_dict=FEATURE_DICT):
    """
    Get list of features from the feature dictionary for snake grammar.
    param: state - current state
    param: feature_dict - dictionary of features.
    """
    values = [Value(state, [i], [CATEGORY_NAMES[j]]) for i,j in feature_dict["Value"]]
    is_equals = [IsEqual(state, [i], [CATEGORY_NAMES[j], CATEGORY_NAMES[k]]) for i,j,k in feature_dict["IsEqual"]]

    # snake grammar is made up of 'Values' and 'IsEquals'
    return values + is_equals

def get_list_of_features_from_dict_traffic(state):
    """
    Get list of features from the feature dictionary for traffic grammar.
    param: state - current state.
    """
    values = []
    diffs = []
    avgs = []

    indices = [(i, j) for i in range(len(state['ACT'])) for j in range(i+1, len(state['ACT']))]

    for categ in state.keys():
        # add value
        values += [traffic_feats.Value(categ, state[categ], i) for i in range(len(state[categ]))]

        # add diffs
        diffs += [traffic_feats.Difference(categ, state[categ], m[0], m[1]) for m in indices]

        # add avgs
        avgs += [traffic_feats.Average(categ, state[categ], list(range(m))) for m in range(1, len(state[categ]))]

    return values + diffs + avgs

def orig_state_action_pair_to_state_dict(state, act):
    """
    Convert state action pair into state dict - which is the middle between 'Learnable state' which we feed into our trees,
    and the original state.
    """
    act = DIR_TO_LIST_DICT[NUM_TO_DIR[act]]
    state_dict = {}
    for category in CATEGORY_NAMES:
        if category == 'ACT':
            state_dict[category] = act
            continue
        start = CATEGORY_DICT[category]*4
        end = start + 4
        state_dict[category] = state[start:end]

    return state_dict

def from_state_dict_to_learnable_state(state_dict, feature_dict=FEATURE_DICT):
    """
    Convert a state_dict to learnable state - with features from the feature_dict.
    param: state_dict - the state dict.
    param: feature_dict - the dictionary of features.
    """
    feature_list = get_list_of_features_from_dict(state_dict, feature_dict)
    learnable_state = []
    for feature in feature_list:
        learnable_state.append(feature.evaluate())
    return learnable_state


def from_state_dict_to_learnable_state_traffic(state_dict, feature_dict=FEATURE_DICT):
    """
    Convert a state_dict to learnable state - with features from the traffic grammar.
    param: state_dict - the state dict.
    param: feature_dict - the dictionary of features.
    """
    feature_list = get_list_of_features_from_dict_traffic(state_dict)
    learnable_state = []
    for feature in feature_list:
        learnable_state.append(feature.evaluate())
    return learnable_state


def is_bad_prop_code_0(d):
    """
    BAD_PROPERTY CODE 0 - if the direction and the apple are equivalent but the action in not in the same direction - this is undesirable
    """
    if not [i for i, e in enumerate(d['DIR']) if e != 0]:
        return False
    index_non_zero_dir = [i for i, e in enumerate(d['DIR']) if e != 0][0]
    index_non_zero_act = [i for i, e in enumerate(d['ACT']) if e != 0][0]
    indices_non_zero_apl = [i for i, e in enumerate(d['APL']) if e != 0]

    if index_non_zero_dir in indices_non_zero_apl:
        if index_non_zero_act != index_non_zero_dir:
            return True

    return False


def is_bad_prop_code_1(d):
    """
    BAD PROPERTY CODE 1 - if the direction and the apple are equivalent but the action in not in the same dir, and the dir is not obs - this is undesirable.
    This is the one we use in our papar.
    """
    if not [i for i, e in enumerate(d['DIR']) if e != 0]:
        return False

    # get indices
    index_non_zero_dir = [i for i, e in enumerate(d['DIR']) if e != 0][0]
    index_non_zero_act = [i for i, e in enumerate(d['ACT']) if e != 0][0]
    indices_non_zero_apl = [i for i, e in enumerate(d['APL']) if e != 0]
    indices_non_zero_obs = [i for i, e in enumerate(d['OBS']) if e != 0]

    # IF direction is to apple
    if index_non_zero_dir in indices_non_zero_apl:
        # and direction is not in obs
        if index_non_zero_dir not in indices_non_zero_obs:
            # and action is not dir
            if index_non_zero_act != index_non_zero_dir:
                return True

    return False

def is_bad_prop(entry, prop_code=0):
    """
    Returns if current state action pair is bad, by the bad property code
    """
    if prop_code == 0:
        return is_bad_prop_code_0(entry)
    if prop_code == 1:
        return is_bad_prop_code_1(entry)

def write_list(l):
    s_start = "["
    s_end = "]\n"

    s = ""

    for i, a in enumerate(l):
        if i != len(l) - 1:
            s = s + str(a.evaluate()) + ", "
        else:
            s = s + str(a.evaluate())
    return s_start + s + s_end

def to_traffic():
    """
    Convert entries with snake features into entries with traffic features.
    """
    orig_des = r'Properties\PROPERTY 1\des.txt'
    orig_undes = r'Properties\PROPERTY 1\undes.txt'

    traffic_des = r'Properties\PROPERTY 1\des_traffic.txt'
    traffic_undes = r'Properties\PROPERTY 1\undes_traffic.txt'

    for l in open(orig_undes, "r"):
        res = l.strip('][\n').split(', ')
        res = [int(en) for en in res[0:17]]
        d = {
            "APL" : res[0:4],
            "OBS" : res[4:8],
            "DIR" : res[8:12],
            "ACT" : res[12:16]
        }

        feats = get_list_of_features_from_dict_traffic(d)

        to_write = write_list(feats)

        with open(traffic_undes, "a+") as f:
            f.write(to_write)

    for l in open(orig_des, "r"):
        res = l.strip('][\n').split(', ')
        res = [int(en) for en in res[0:17]]
        d = {
            "APL" : res[0:4],
            "OBS" : res[4:8],
            "DIR" : res[8:12],
            "ACT" : res[12:16]
        }

        feats = get_list_of_features_from_dict_traffic(d)

        to_write = write_list(feats)

        with open(traffic_des, "a+") as f:
            f.write(to_write)