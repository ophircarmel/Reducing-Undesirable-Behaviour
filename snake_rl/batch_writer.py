running_txt_train = r"PTH_TO_PYTHON PTH_TO_agent1.py -t 0"
running_txt_test = r"PTH_TO_PYTHON PTH_TO_agent1.py -t 1"


starting_arr = 900

reward_modifs = [1.0, 0.1]

iters_per_modif = 5

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
            print(running_txt_test + " -ru " + str(run_number) + "\n")

def main():
    print('\n\nTrains:')
    train()
    print("\n\nTests:")
    test()

if __name__ == '__main__':
    main()