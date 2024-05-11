"""
Utility function for classifying and labeling
"""
from TrafficControl.TLCS.Tree.features import Value, Difference, Average
from TrafficControl.TLCS.Tree.snake_features import Value as Snake_Val, IsEqual

STATE_LENGTH = 80
LANES = 8
CELLS_PER_LANE = int(STATE_LENGTH / LANES)
FIRST_N = 8
BAD_PROP_CODE = 0



# the dictionary of all our features
FEATURE_DICT = {
    "Value" : [i for i in range(FIRST_N)],
    "Difference" : [(0, i) for i in range(1, FIRST_N)],
    "Average" : [[j for j in range(i)] for i in range(3, FIRST_N)]
}


# get list of features from the feature dictionary
def get_list_of_features_from_dict(lane_ID, lane, feature_dict):
    """
    Get the list of features from using the original grammar.
    param: lane_ID - which lane
    param: lane - values of lane
    param: feature_dict - the grammar
    """
    values = [Value(lane_ID, lane, i) for i in feature_dict["Value"]]
    differences = [Difference(lane_ID, lane, i[0], i[1]) for i in feature_dict["Difference"]]
    averages = [Average(lane_ID, lane, i) for i in feature_dict["Average"]]

    return values + differences + averages

def get_list_of_features_from_dict_snake(d):
    """
    Get the list of features from using 'snake' grammar.
    param: d - the state-action pair
    """

    values = [Snake_Val(lane_ID=id, lane=d[id], index=idx) for id in d.keys() for idx in range(len(d[id]))]

    idss = list(d.keys())
    idss.remove("ACT")
    all_ids = [(idss[i], idss[j]) for i in range(len(idss)) for j in range(i+1, len(idss))]

    isequals = [IsEqual(lane_ID=idd[0], lane=d[idd[0]], other_lane_ID=idd[1], other_lane=d[idd[1]], index=idx) for idd in all_ids for idx in range(len(d[idd[0]]))]

    # snake grammar is made up of values and isequals
    return values + isequals


def get_data_from_state(state, first_n):
    """
    State is a list with 80 entries.
    Return a dictionary of 8 lists:
        Each list is the queue in a specific lane, until with maximal queue size of first_n
    Index 0 - W2E
    Index 1 - W2TL
    Index 4 - E2W
    Index 5 - E2TL
    Index 2 - N2S
    Index 3 - N2TL
    Index 6 - S2N
    Index 7 - S2TL
    """
    l = []
    n = 10
    state = state.tolist()
    for i in range(0, len(state), n):
        part = state[i:i + n]
        l.append(part[:first_n])
    d = {
        "W2E": l[0], "W2TL": l[1],
        "E2W": l[4], "E2TL": l[5],
        "N2S": l[2], "N2TL": l[3],
        "S2N": l[6], "S2TL": l[7]
    }
    return d


def get_data_from_action(action):
    """
    Action Number to lane.
    param: action - the action
    """
    if action == 0:
        act = "NS"
    elif action == 1:
        act = "NSL"
    elif action == 2:
        act = "EW"
    elif action == 3:
        act = "EWL"
    return act



def get_state_action_dictionary(state, action, first_n):
    """
    Return state and action dictionary from state and action.
    param: state - current state.
    param: action - the action we took.
    param: first_n - how much of the lane should we take.
    """
    # get state
    d = get_data_from_state(state, first_n)
    # get action
    d["ACT"] = get_data_from_action(action)
    return d


def sum_consecutive(l):
    """
    Sum consecutive cells in a list
    """
    sum = 0
    j = 0
    while l[j] > 0:
        sum += 1
        j += 1
        if j == len(l):
            break
    return sum

def is_bad_prop_code_0(d):
    """
    BAD_PROPERTY CODE 0 - there are jams (queue > 1) and the action is not taking care of any of the jams,
    this is undesirable.
    param: d - the state-action dictionary
    returns: 1 if desirable, 0 is undesirable
    """
    # s counts for each lane, how many cells in row is the queue
    s = {}
    try:
        s["W2E"] = sum_consecutive(d["W2E"])
        s["W2TL"] = sum_consecutive(d["W2TL"])
        s["E2W"] = sum_consecutive(d["E2W"])
        s["E2TL"] = sum_consecutive(d["E2TL"])
        s["N2S"] = sum_consecutive(d["N2S"])
        s["N2TL"] = sum_consecutive(d["N2TL"])
        s["S2N"] = sum_consecutive(d["S2N"])
        s["S2TL"] = sum_consecutive(d["S2TL"])
        s["ACT"] = d["ACT"]
    except Exception:
        x = 1
    bad_state = True

    # each of those is a boolean variable which determines if lane is jammed
    ewj = ((s["W2E"] > 1) or (s["E2W"] > 1))
    ewlj = ((s["W2TL"] > 1) or (s["E2TL"] > 1))
    nsj = ((s["N2S"] > 1) or (s["S2N"] > 1))
    nslj = ((s["N2TL"] > 1) or (s["S2TL"] > 1))

    # No jam
    if (not ewj) and (not nsj) and (not ewlj) and (not nslj):
        bad_state = False

    # Jam but the action is directly assisting the removal of said jam
    if ewj and s["ACT"] == "EW":
        bad_state = False
    if nsj and s["ACT"] == "NS":
        bad_state = False
    if ewlj and s["ACT"] == "EWL":
        bad_state = False
    if nslj and s["ACT"] == "NSL":
        bad_state = False

    # bad state is true only if there are jams and the action is not assisting the removal of any jam
    return bad_state


def is_bad_prop(entry, prop_code):
    """
    Returns if current state action pair is undesirable, by the bad property code
    """
    if prop_code == 0:
        return is_bad_prop_code_0(entry)