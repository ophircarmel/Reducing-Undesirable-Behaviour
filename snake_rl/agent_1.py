import argparse
import datetime
import json
import os
import pickle
import sys
import h5py

import wandb

from Tree.utils_for_tree import *
from snake_env import Snake

import random
import numpy as np
from keras import Sequential
from collections import deque
from keras.layers import Dense
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from plot_script import plot_result
import time


#######################
# START ADDITION
#######################
# if we want to run with the traffic grammar - turn this to 'True'
TRAFFIC = False

UNDES_FILE = "Tree/Properties/PROPERTY 1/undes.txt"
DES_FILE = "Tree/Properties/PROPERTY 1/des.txt"

if TRAFFIC:
    UNDES_FILE_TRAFFIC = "Tree/Properties/PROPERTY 0/undes_traffic.txt"
    DES_FILE_TRAFFIC = "Tree/Properties/PROPERTY 0/des_traffic.txt"
#######################
# END ADDITION
#######################

class DQN:
    """ Deep Q Network """

    def __init__(self, env, params):

        self.action_space = env.action_space
        self.state_space = env.state_space
        self.epsilon = params['epsilon']
        self.gamma = params['gamma']
        self.batch_size = params['batch_size']
        self.epsilon_min = params['epsilon_min']
        self.epsilon_decay = params['epsilon_decay']
        self.learning_rate = params['learning_rate']
        self.layer_sizes = params['layer_sizes']
        self.memory = deque(maxlen=2500)
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        for i in range(len(self.layer_sizes)):
            if i == 0:
                model.add(Dense(self.layer_sizes[i], input_shape=(self.state_space,), activation='relu'))
            else:
                model.add(Dense(self.layer_sizes[i], activation='relu'))
        model.add(Dense(self.action_space, activation='softmax'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):

        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self):

        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states = np.array([i[0] for i in minibatch])
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        next_states = np.array([i[3] for i in minibatch])
        dones = np.array([i[4] for i in minibatch])

        states = np.squeeze(states)
        next_states = np.squeeze(next_states)

        targets = rewards + self.gamma * (np.amax(self.model.predict_on_batch(next_states), axis=1)) * (1 - dones)
        targets_full = self.model.predict_on_batch(states)

        ind = np.array([i for i in range(self.batch_size)])
        targets_full[[ind], [actions]] = targets

        self.model.fit(states, targets_full, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def train_dqn(episode, env):
    sum_of_rewards = []
    agent = DQN(env, params)
    undesirable_lst = []
    desirable_lst = []
    for e in range(episode):
        state = env.reset()
        state = np.reshape(state, (1, env.state_space))
        score = 0
        max_steps = 1000
        undesirable_cntr = 0
        desirable_cntr = 0
        for i in range(max_steps):
            print(i)
            action = agent.act(state)

            #######################
            # START ADDITION
            #######################
            # d is the 'state dict' - an aggregation of the 'state' and 'action'
            # we can later evaluate whether it is 'Desirable' or 'Undesirable'.
            d = orig_state_action_pair_to_state_dict(state[0].tolist(), action)
            # l is in the format that the tree can learn on later
            l = from_state_dict_to_learnable_state(d)
            # if we use traffic grammar
            l_traffic = from_state_dict_to_learnable_state_traffic(d)

            # if this isn't the first step
            if d['DIR'] != [0, 0, 0, 0]:
                if is_bad_prop(d):
                    # if the state-action pair is undesirable
                    with open(UNDES_FILE, "a+") as f:
                        f.write(str(l))
                        f.write("\n")

                    with open(UNDES_FILE_TRAFFIC, "a+") as f:
                        f.write(str(l_traffic))
                        f.write("\n")

                    undesirable_cntr += 1
                else:
                    # if the state-action pair is desirable
                    with open(DES_FILE, "a+") as f:
                        f.write(str(l))
                        f.write("\n")

                    with open(DES_FILE_TRAFFIC, "a+") as f:
                        f.write(str(l_traffic))
                        f.write("\n")

                    desirable_cntr += 1

            #######################
            # END ADDITION
            #######################

            # print(action)
            prev_state = state
            next_state, reward, done, _ = env.step(action)
            score += reward
            next_state = np.reshape(next_state, (1, env.state_space))
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if params['batch_size'] > 1:
                agent.replay()
            if done:
                print(f'final state before dying: {str(prev_state)}')
                print(f'episode: {e + 1}/{episode}, score: {score}')
                #######################
                # START ADDITION
                #######################
                print("Undesirable: " + str(undesirable_cntr), ", Desirable: " + str(desirable_cntr))
                undesirable_lst.append(undesirable_cntr)
                desirable_lst.append(desirable_cntr)
                #######################
                # END ADDITION
                #######################
                break
        # with open('agent.pkl', 'wb+') as f:
        #     pickle.dump(agent, f)
        #     print("DUMPED")
        sum_of_rewards.append(score)
        #######################
        # START ADDITION
        #######################
        # logging
        print("\nUNDESIRABLE_LST: " + str(undesirable_lst) + "\n")
        print("DESIRABLE_LST: " + str(desirable_lst) + "\n")
        print("UNDESIRABLE / EVERYTHING: " + str(
            [undesirable_lst[i] / (desirable_lst[i] + undesirable_lst[i]) for i in range(len(undesirable_lst))]) + "\n")
        #######################
        # END ADDITION
        #######################
    return sum_of_rewards


def run_model(episode, env, dt, model_pth):
    #######################
    # START ADDITION
    #######################
    # get relevant model
    with open(model_pth + "episode-49-agent.pkl", 'rb') as f:
        agent = pickle.load(f)
    #######################
    # END ADDITION
    #######################

    sum_of_rewards = []
    ratios = []
    for e in range(episode):

        state = env.reset()
        state = np.reshape(state, (1, env.state_space))
        score = 0
        max_steps = 10000
        undesirable_cntr = 0
        desirable_cntr = 0

        for i in range(max_steps):
            action = agent.act(state)

            #######################
            # START ADDITION
            #######################

            # get state_action pair
            d = orig_state_action_pair_to_state_dict(state[0].tolist(), action)
            # get true label
            label = is_bad_prop(d, prop_code=1)

            # add to relevant cntr
            if d['DIR'] != [0, 0, 0, 0]:
                if label:
                    undesirable_cntr += 1
                else:
                    desirable_cntr += 1
            #######################
            # END ADDITION
            #######################

            prev_state = state
            next_state, reward, done, _ = env.step(action)
            score += reward
            next_state = np.reshape(next_state, (1, env.state_space))
            state = next_state
            if done:
                print(f'final state before dying: {str(prev_state)}')
                print(f'episode: {e + 1}/{episode}, score: {score}')
                break

        #######################
        # START ADDITION
        #######################
        # logging
        ratio = undesirable_cntr / (desirable_cntr + undesirable_cntr)
        sum_of_rewards.append(score)
        ratios.append(round(ratio, 4))
        with open(model_pth + 'test_stats_new.txt', 'w+') as f:
            json.dump({'sum_of_rewards': sum_of_rewards, 'Ratio': ratios},
                      f)
            f.write("\n")
            print("DUMPED")
        #######################
        # END ADDITION
        #######################

    return score


def train_model_with_dt(episode, env, dt, reward_modifier, pth_to_model, params):
    """
    Train the Snake agent with the decision tree.
    param: episode - the current episode
    param: env - the environment
    param: dt - the decision tree
    param: reward_modifier - the number to multiply the reward by
    param: pth_to_model - where to save the model
    param: params - parameters for training
    """
    sum_of_rewards = []
    agent = DQN(env, params)
    #######################
    # START ADDITION
    #######################
    # initializations
    undesirable_lst = []
    desirable_lst = []
    ratio_lst = []
    timestamp_start = datetime.datetime.now()
    #######################
    # END ADDITION
    #######################
    for e in range(episode):
        state = env.reset()
        state = np.reshape(state, (1, env.state_space))
        score = 0
        max_steps = 1000
        undesirable_cntr = 0
        desirable_cntr = 0
        for i in range(max_steps):
            print(i)
            action = agent.act(state)

            #######################
            # START ADDITION
            #######################
            # if we put a reward modifier
            if reward_modifier != 1.0:
                # d is the 'state dict' - an aggregation of the 'state' and 'action'
                # we can later evaluate whether it is 'Desirable' or 'Undesirable'.
                d = orig_state_action_pair_to_state_dict(state[0].tolist(), action)
                # extract features from state-action pair to feed into the tree
                l = from_state_dict_to_learnable_state(d)
                # classify 1 or 0
                label = classify(dt, l)
                # if this isn't the first step
                if d['DIR'] != [0, 0, 0, 0]:
                    if label == 0:
                        # undesirable
                        undesirable_cntr += 1
                    else:
                        # desirable
                        desirable_cntr += 1
            #######################
            # END ADDITION
            #######################

            # print(action)
            prev_state = state
            next_state, reward, done, _ = env.step(action)
            pre_reward = reward

            #######################
            # START ADDITION
            #######################
            # this is the most important part - updating the reward
            if reward_modifier != 1.0:
                if d['DIR'] != [0, 0, 0, 0]:
                    # modify the reward
                    reward = modify_reward(label, reward, reward_modifier)
            post_reward = reward
            #######################
            # END ADDITION
            #######################

            score += reward
            next_state = np.reshape(next_state, (1, env.state_space))
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if params['batch_size'] > 1:
                agent.replay()
            if done:
                print(f'final state before dying: {str(prev_state)}')
                print(f'episode: {e + 1}/{episode}, score: {score}')

                #######################
                # START ADDITION
                #######################
                if reward_modifier != 1.0:
                    # logging
                    print("Undesirable: " + str(undesirable_cntr), ", Desirable: " + str(desirable_cntr))
                    undesirable_lst.append(undesirable_cntr)
                    desirable_lst.append(desirable_cntr)
                    ratio_lst.append(undesirable_cntr / (undesirable_cntr + desirable_cntr))

                    # log with wandb
                    wandb.log(
                        {
                            "Reward": score,
                            "Undesirable Ratio": undesirable_cntr / (undesirable_cntr + desirable_cntr)
                        }
                    )
                    #######################
                    # END ADDITION
                    #######################

                break

        #######################
        # START ADDITION
        #######################
        # more logging
        print("\nUNDESIRABLE_LST: " + str(undesirable_lst) + "\n")
        print("DESIRABLE_LST: " + str(desirable_lst) + "\n")
        print("UNDESIRABLE / EVERYTHING: " + str(
            [undesirable_lst[i] / (desirable_lst[i] + undesirable_lst[i]) for i in range(len(undesirable_lst))]) + "\n")

        #######################
        # END ADDITION
        #######################

        # graph_undesirable(desirable_lst, undesirable_lst, reward_modifier)
        sum_of_rewards.append(score)

        #######################
        # START ADDITION
        #######################
        # save model
        with open(pth_to_model + 'episode-' + str(e) + '-agent.pkl', 'wb+') as f:
            pickle.dump(agent, f)
            print("DUMPED")
        with open(pth_to_model + 'stats.txt', 'w+') as f:
            json.dump({'sum_of_rewards': sum_of_rewards, 'desirable': desirable_lst, 'undesirable': undesirable_lst,
                       'ratio': ratio_lst},
                      f)
            f.write("\n")
            print("DUMPED")
        #######################
        # END ADDITION
        #######################

        # with open('agent-' + str(reward_modifier) + '.pkl', 'wb+') as f:
        #     pickle.dump(agent, f)
        #     print("DUMPED")
        # with open('stats-' + str(reward_modifier) + '.pkl', 'wb+') as f:
        #     pickle.dump(
        #         {'sum_of_rewards' : sum_of_rewards, 'desirable' : desirable_lst, 'undesirable' : undesirable_lst},
        #         f
        #     )
    #######################
    # START ADDITION
    #######################
    # log time
    timestamp_end = datetime.datetime.now()

    with open(pth_to_model + 'time.txt', 'w+') as f:
        f.write(str(timestamp_end - timestamp_start))

    #######################
    # END ADDITION
    #######################

    return sum_of_rewards


#######################
# START ADDITION
#######################

def classify(tree, sample):
    """
    Classify a sample to be desirable or undesirable.
    If the sample is desirable - 1
    If the sample is undesirable - 0

    param: tree - the decision tree
    param: sample - the sample we want to evaluate
    returns: classification of the sample
    """
    r = tree.predict([np.array(sample).astype(np.float32)])[0]
    return int(r)

def modify_reward(label, reward, reward_modifier):
    """
    Modify the reward.
    The reward is < 0 usually, so reward_modifier is > 1.
    If the reward is < 0, then we multiply by reward_modifier
    If the reward is > 0, then we multiply by 1/reward_modifier
    param: label - 1 for desirable, 0 for undesirable
    param: reward - the original reward
    param: reward modifier - the relevant reward modifier
    return: updated reward
    """
    # If the state-action pair is a-okay
    if label != 0:
        return reward
    # If the state-action pair is problematic
    if reward < 0:
        return reward * 1 / reward_modifier
    return reward * reward_modifier


def graph_undesirable(des, undes, reward_modifier):
    """
    Graph the desirable and undesirable ratio.
    """
    des_over_tot = [des[i] / (des[i] + undes[i]) for i in range(len(des))]
    undes_over_tot = [undes[i] / (des[i] + undes[i]) for i in range(len(undes))]
    plt.plot(des_over_tot, label='DESIRABLE_OVER_TOTAL')
    plt.plot(undes_over_tot, label='UNDESIRABLE_OVER_TOTAL')
    plt.title('Desirable and Undesirable over total, for reward modifier = ' + str(reward_modifier))
    plt.legend()
    plt.show()


#######################
# END ADDITION
#######################

def test_run(run, env):
    # load decision tree
    loaded_model = pickle.load(open('Tree/Properties/PROPERTY 1/new_tree.pkl', 'rb'))

    # init params
    params = dict()
    params['name'] = None
    params['epsilon'] = 1
    params['gamma'] = .95
    params['batch_size'] = 500
    params['epsilon_min'] = .01
    params['epsilon_decay'] = .995
    params['learning_rate'] = 0.00025
    params['layer_sizes'] = [128, 128, 128]

    ep = 50

    # get saved model
    # you may need to change this path in order to save correctly
    model_pth = r"snake_rl\models\model_" + str(run) + "/"

    # run and evaluate
    run_model(ep, env, loaded_model, model_pth)

    # close environment
    env.bye()


def train_with_modifier(run, modifier):
    # initialize wandb - if you don't have one - remove this part
    wandb.init(
        project="Snake",
        entity="username",
        name='Run  ' + str(run) + ' - Modifier = ' + str(modifier),
        save_code=True,
        tags=[str(modifier)],
        id='run' + str(run),
        # config=config
    )

    # load the tree for snake
    loaded_model = pickle.load(open('Tree/Properties/PROPERTY 1/new_tree.pkl', 'rb'))

    # initialize parameters
    params = dict()
    params['name'] = None
    params['epsilon'] = 1
    params['gamma'] = .95
    params['batch_size'] = 500
    params['epsilon_min'] = .01
    params['epsilon_decay'] = .995
    params['learning_rate'] = 0.00025
    params['layer_sizes'] = [128, 128, 128]

    # init env
    env = Snake()

    ep = 50

    # save model

    # you may need to change this path in order to save correctly
    model_pth = r"snake_rl\models\model_" + str(run) + "/"
    if not os.path.isdir(model_pth):
        os.mkdir("models/model_" + str(run))

    sum_of_rewards = train_model_with_dt(ep, env, loaded_model, reward_modifier=modifier, pth_to_model=model_pth,
                                         params=params)

    # close env
    env.bye()

    results = {}
    results[params['name']] = sum_of_rewards

    # plot_result(results, direct=True, k=20)


def train_much(to_train):
    for mod in to_train.keys():
        for mod_no in to_train[mod]:
            print("MODEL TRAIN: " + str(mod_no))
            train_with_modifier(mod_no, float(mod))


def test_much(to_test):
    for mod in to_test.keys():
        for mod_no in to_test[mod]:
            print("MODEL TEST: " + str(mod_no))
            test_run(mod_no, Snake())


if __name__ == '__main__':
    # if input('Input y for run model\n') == 'y':
    #     run_model(50, Snake())
    #     quit()

    # is_test = int(sys.argv[1])
    # mod_num = int(sys.argv[2])
    #
    # if is_test == 0:
    #     modif = float(sys.argv[3])
    #     print("MODEL TRAIN: " + str(mod_num))
    #     train_with_modifier(mod_num, float(modif))
    #
    # else:
    #     print("MODEL TEST: " + str(mod_num))
    #     test_run(mod_num, Snake())
    #
    # exit(0)

    # to = {
    #     1 : [400, 401, 402, 403],
    #     0.75 : [410, 411, 412, 413],
    #     0.5 : [420, 421, 422, 423],
    #     0.25 : [430, 431, 432, 433],
    #     0.1 : [440, 441, 442, 443],
    # }
    #
    # train_much(to)
    # test_much(to)
    # exit(0)

    # Construct the argument parser
    ap = argparse.ArgumentParser()

    # Add the arguments to the parser
    ap.add_argument("-ru", "--run_number", required=True,
                    help="Run Number")
    ap.add_argument("-re", "--reward_modifier", required=False,
                    help="Reward Modifier")
    ap.add_argument("-t", "--train_or_test", required=True, help="Train or Test")
    args = vars(ap.parse_args())

    if int(args["train_or_test"]) == 0:
        # training the model
        train_with_modifier(int(args["run_number"]), float(args["reward_modifier"]))
    else:
        # testing the model
        test_run(int(args["run_number"]))

    exit(0)

    params = dict()
    params['name'] = None
    params['epsilon'] = 1
    params['gamma'] = .95
    params['batch_size'] = 500
    params['epsilon_min'] = .01
    params['epsilon_decay'] = .995
    params['learning_rate'] = 0.00025
    params['layer_sizes'] = [128, 128, 128]

    results = dict()
    ep = 50

    # for batchsz in [1, 10, 100, 1000]:
    #     print(batchsz)
    #     params['batch_size'] = batchsz
    #     nm = ''
    #     params['name'] = f'Batchsize {batchsz}'
    env_infos = {'States: only walls': {'state_space': 'no body knowledge'},
                 'States: direction 0 or 1': {'state_space': ''}, 'States: coordinates': {'state_space': 'coordinates'},
                 'States: no direction': {'state_space': 'no direction'}}

    # for key in env_infos.keys():
    #     params['name'] = key
    #     env_info = env_infos[key]
    #     print(env_info)
    #     env = Snake(env_info=env_info)
    env = Snake()
    # loaded_model = pickle.load(open('Tree/Properties/PROPERTY 0/finalized_tree.pkl', 'rb'))
    # sum_of_rewards = train_model_with_dt(ep, env, loaded_model, reward_modifier=float(sys.argv[1]))

    sum_of_rewards = train_dqn(ep, env)

    results[params['name']] = sum_of_rewards

    plot_result(results, direct=True, k=20)
