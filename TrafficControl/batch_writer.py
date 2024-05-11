running_txt_train = r"path_to_python TLCS/training_main.py"
running_txt_test = r"path_to_python TLCS/test_multiple.py"

starting_arr = 400

reward_modifs = [1.0, 0.75, 0.5, 0.25, 0.1]

iters_per_modif = 3

def train():
    """
    Lines for running training on cluster
    """
    for i, modif in enumerate(reward_modifs):
        for j in list(range(iters_per_modif)):
            reward_modifier = modif
            run_number = starting_arr + i*10 + j
            print(running_txt_train + " -ru " + str(run_number) + " -re " + str(reward_modifier) + "\n")


def test():
    """
    Lines for running testing on cluster
    """
    for i, modif in enumerate(reward_modifs):
        for j in list(range(iters_per_modif)):
            reward_modifier = modif
            run_number = starting_arr + i*10 + j
            print(running_txt_test + " " + str(run_number) + "\n")

# write train lines followed by test lines.
train()
test()