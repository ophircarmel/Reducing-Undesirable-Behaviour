"""
Generate a random input array
"""
import numpy as np

# total length of init choice array
TOTAL_INIT = 100

# probability of each inital number
INITIAL_TEN_PROBA = 0.14
INITIAL_ELEVEN_PROBA = 0.17
INITIAL_TWELVE_PROBA = 0.17
INITIAL_THIRTEEN_PROBA = 0.15
INITIAL_FOURTEEN_PROBA = 0.12
INITIAL_FIFTEEN_PROBA = 0.1
INITIAL_SIXTEEN_PROBA = 0.07
INITIAL_SEVENTEEN_PROBA = 0.04
INITIAL_EIGHTEEN_PROBA = 0.02
INITIAL_NINETEEN_PROBA = 0.01
INITIAL_TWENTY_PROBA = 0.01

INIT_ARR = [10] * int(TOTAL_INIT * INITIAL_TEN_PROBA) + \
           [11] * int(TOTAL_INIT * INITIAL_ELEVEN_PROBA) + \
           [12] * int(TOTAL_INIT * INITIAL_TWELVE_PROBA) + \
           [13] * int(TOTAL_INIT * INITIAL_THIRTEEN_PROBA) + \
           [14] * int(TOTAL_INIT * INITIAL_FOURTEEN_PROBA) + \
           [15] * int(TOTAL_INIT * INITIAL_FIFTEEN_PROBA) + \
           [16] * int(TOTAL_INIT * INITIAL_SIXTEEN_PROBA) + \
           [17] * int(TOTAL_INIT * INITIAL_SEVENTEEN_PROBA) + \
           [18] * int(TOTAL_INIT * INITIAL_EIGHTEEN_PROBA) + \
           [19] * int(TOTAL_INIT * INITIAL_NINETEEN_PROBA) + \
           [20] * int(TOTAL_INIT * INITIAL_TWENTY_PROBA)

# total choice size
DELTA_TOT = 100

# probability of each delta
DELTA_0 = 0.25
DELTA_1 = 0.3
DELTA_2 = 0.25
DELTA_3 = 0.13
DELTA_4 = 0.07

DELTA_ARR = [0] * int(DELTA_TOT * DELTA_0) + \
            [1] * int(DELTA_TOT * DELTA_1) + \
            [2] * int(DELTA_TOT * DELTA_2) + \
            [3] * int(DELTA_TOT * DELTA_3) + \
            [4] * int(DELTA_TOT * DELTA_4)

"""
Generate random initial entry
"""


def init_number():
    init_num = np.random.choice(INIT_ARR)
    return init_num


"""
Generate random next entry by previous entry
"""


def next_number_by_previous(prev_num):
    # get random delta
    delta = np.random.choice(DELTA_ARR)

    # adjust probability of adding a positive or negative number by bounds
    if prev_num < 11:
        proba_pos = 0.3
    elif prev_num > 20:
        proba_pos = 0.7
    else:
        proba_pos = 0.55

    # random decision if we add a positive delta or a negative delta
    if np.random.random() >= proba_pos:
        posneg = 1
    else:
        posneg = -1

    # calculate the next entry
    delta = posneg * delta
    next_num = prev_num + delta

    # bounds
    if next_num > 24:
        next_num = 24
    if next_num < 10:
        next_num = 10

    return next_num


"""
Generate a random array
"""


def generate_random_arr():
    # random initial number
    init_num = init_number()
    rand_arr = [init_num]

    for i in range(9):
        # add random next number
        rand_arr.append(next_number_by_previous(rand_arr[-1]))
    return rand_arr


"""
Generator for a random array
"""


def random_array_generator():
    while True:
        yield generate_random_arr()
