import json

from TrafficControl.TLCS.utils_for_tree import is_bad_prop, get_list_of_features_from_dict, FEATURE_DICT, LANES, \
    get_list_of_features_from_dict_snake
from TrafficControl.TLCS.consts import *

# read data from file and write into input file
def read_data_file_write_with_features(data_file):
    hand = open(data_file, "r")

    line = hand.readline()

    while line:
        # if the line is "\n"
        if len(line) < 5:
            line = hand.readline()
            continue

        # do one entry
        entry_dict = json.loads(line)

        # determine label
        if is_bad_prop(entry_dict, 0):
            label = 0
        else:
            label = 1

        # create features
        if SNAKE:
            action = entry_dict["ACT"]
            if action == "NS":
                act = [1, 0, 0, 0]
            elif action == "NSL":
                act = [0, 1, 0, 0]
            elif action == "EW":
                act = [0, 0, 1, 0]
            elif action == "EWL":
                act = [0, 0, 0, 1]
            entry_dict["ACT"] = act

            entry_feats = get_list_of_features_from_dict_snake(entry_dict)
        else:
            entry_feats = []
            for key in entry_dict.keys():
                if key == "ACT":
                    action = entry_dict[key]
                    if action == "NS":
                        act = [1, 0, 0, 0]
                    elif action == "NSL":
                        act = [0, 1, 0, 0]
                    elif action == "EW":
                        act = [0, 0, 1, 0]
                    elif action == "EWL":
                        act = [0, 0, 0, 1]
                    continue
                lane_ID = key
                lane = entry_dict[key]
                entry_feats += get_list_of_features_from_dict(lane_ID, lane, FEATURE_DICT)

        # write entry to new file
        if label == 0:
            f = open(UNDES_FILE, "a+")
        elif label == 1:
            f = open(DES_FILE, "a+")

        evals_str = []
        for feat in entry_feats:
            evals_str.append(str(round(feat.evaluate(), 2)))
        jmp = int(len(evals_str) / LANES)
        print("Writing")
        f.write("[\n")
        if SNAKE:
            f.write(", ".join(evals_str) + "\n")
        else:
            for i in range(LANES):
                f.write(" ,".join(evals_str[i*jmp:(i+1)*jmp]) + "\n")
            f.write(", ".join([str(ac) for ac in act]) + "\n")
        f.write("]\n\n")
        #f.write("]
        # \nLabel = " + str(label) + "\n\n")

        line = hand.readline()


# read data from file and write into input file
def read_data_file_write_no_features(data_file):
    hand = open(data_file, "r")

    line = hand.readline()

    while line:
        # if the line is "\n"
        if len(line) < 5:
            line = hand.readline()
            continue

        # do one entry
        entry_dict = json.loads(line)

        # determine label
        if is_bad_prop(entry_dict, 0):
            label = 0
        else:
            label = 1

        action = entry_dict["ACT"]
        if action == "NS":
            act = [1, 0, 0, 0]
        elif action == "NSL":
            act = [0, 1, 0, 0]
        elif action == "EW":
            act = [0, 0, 1, 0]
        elif action == "EWL":
            act = [0, 0, 0, 1]
        entry_dict["ACT"] = act

        # write entry to new file
        if label == 0:
            f = open(UNDES_FILE, "a+")
        elif label == 1:
            f = open(DES_FILE, "a+")

        print("Writing")
        f.write("[\n")
        for key in entry_dict:

            f.write(" ,".join([str(e) for e in entry_dict[key]]) + "\n")
        f.write("]\n\n")

        line = hand.readline()



if __name__ == '__main__':
    read_data_file_write_with_features("../training_file.txt")








