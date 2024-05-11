import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import wandb

SMOOTHING = 0.01
WINDOW = 250

api = wandb.Api(timeout=20)

def numpy_ewma_vectorized_v2(data, window):
    """
    Obsolete function
    """
    alpha = 2 / (window + 1.0)
    alpha_rev = 1 - alpha
    n = data.shape[0]

    pows = alpha_rev ** (np.arange(n + 1))

    scale_arr = 1 / pows[:-1]
    offset = data[0] * pows[1:]
    pw0 = alpha * alpha_rev ** (n - 1)

    mult = data * pw0 * scale_arr
    cumsums = mult.cumsum()
    out = offset + cumsums * scale_arr[::-1]
    return out

def combine_steps(steps_arrays):
    """
    Combine a list of lists into a big list and drop duplicates.
    param: steps_arrays - the list of lists
    """
    all_steps = []
    for steps in steps_arrays:
        all_steps += steps
    return sorted(list(set(all_steps)))


def running_mean(x, N):
    """
    Get running mean of x over a window of N.
    param: x - the list
    param: N - the window size
    """
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


def plot_avg(run_ids, param):
    """
    Plot convergence over some run numbers.
    param: run_ids - a list of lists, we want to plot the average defined metric for
                     each sublist
    param: param - the metric we want to plot
    """
    all_step_param_pairs = []

    steps_arrays = []

    for run_id in run_ids:
        # get run data
        try:
            pth = "wandb_runs/" + run_id
            if os.path.exists(pth + "_hist"):
                run_data = pickle.load(open(pth + "_hist", "rb"))
                conf = pickle.load(open(pth + "_conf", "rb"))
            else:
                # shouldn't happen - I added all the relevant models
                # you can comment this part out
                entity_name = "name_of_entity_here"
                run = api.run(entity_name + "/PCC-RL-GRAPHING/" + run_id)
                pickle.dump(run.history(samples=2000), open("wandb_runs/" + run_id + "_hist", "wb+"))
                pickle.dump(run.config, open("wandb_runs/" + run_id + "_conf", "wb+"))
                run_data = run.history(samples=2000)
                conf = run.config

            # get reward_modifier
            downscaling = conf['alpha_downscaling']

            print(run_id)

            steps = run_data['_step'].to_numpy().tolist()

            steps_arrays.append(steps)

            param_data = run_data[param].to_numpy().tolist()

            step_param_pairs = {steps[i]: param_data[i] for i in range(len(steps))}

            # get the pairs of metric and each one of the runs
            all_step_param_pairs.append(step_param_pairs)

        except Exception as e:
            print(e)
            print("Exclude " + str(run_id))
            continue

    all_steps = combine_steps(steps_arrays)
    avg_arr = []

    # get average of each run group
    for step in all_steps:
        sum_curr_step = 0
        summed_curr_step = 0
        for step_param_pair in all_step_param_pairs:
            try:
                sum_curr_step += step_param_pair[step]
                summed_curr_step += 1
            except Exception:
                continue

        avg_arr.append(sum_curr_step / summed_curr_step)

    # plot the running mean convergence
    plt.plot(all_steps[WINDOW-1:], running_mean(avg_arr, WINDOW), label='Reward Modifier=' + str(downscaling))
    #plt.plot(all_steps, [np.mean(avg_arr[600:])]*len(all_steps), label='Reward Modifier=' + str(downscaling))

    plt.xlabel('step')
    plt.ylabel('Value')
    print('Downscaling = ' + str(downscaling))
    #print(running_mean(running_mean(avg_arr, WINDOW), 50)[-1])
    #print(np.mean(avg_arr[600:]))
    return all_steps, avg_arr, downscaling


def get_time_by_model(run_number):
    timestamp_file = r"src\gym\runs\runs21" + str(run_number) + r"\model\timestamp.txt"
    with open(timestamp_file, "r") as f:
        time_str = f.readline().strip()
        time_str = time_str.split('.')[0]
        hours, minutes, seconds = time_str.split(':')

        num_of_mins = int(hours) * 60 + int(minutes) + round(int(seconds) / 60)
    return num_of_mins

def get_times_by_models(run_numbers):
    timesss = []
    for nums in run_numbers:
        timess = []
        for run in nums:
            t = get_time_by_model(run)

            # outliers
            if t > 400:
                continue
            timess.append(t)
        print(timess)
        timesss.append(np.mean(timess))
    print(timesss[0])
    print(np.mean(timesss[1:]))
    print("Original model's time / Model with tree's time: " + str(timesss[0] / np.mean(timesss[1:])) + "\n")


def compare_avgs_of_runs():
    """
    Getting the graphs we displayed in the paper:
    Comparing the undesirable behavior ratio and average ewma reward for different reward_modifiers
    """
    batch_id = 'tst20'

    IDSSSS = [
        ['00', '01', '02', '03', '04'],
        ['10', '11', '12', '13', '14'],
        ['20', '21', '22', '23', '24'],
        ['30', '31', '32', '33', '34'],
        ['40', '41', '42', '43', '44'],
        ['50', '51', '52', '53', '54'],
        ['60', '61', '62', '63', '64'],
        ['70', '71', '72', '73', '74'],
        ['80', '81', '82', '83', '84'],
    ]

    # runs that failed
    EXCLUDE = ["54", "80", "11"]

    # get relevant run numbers
    NEWIDS = [[batch_id + IDSSSS[i][j] for j in range(len(IDSSSS[i])) if IDSSSS[i][j] not in EXCLUDE] for i in
              range(len(IDSSSS))]
    IDS = NEWIDS

    print(IDS)

    # params = ['Ewma', 'Undesirable to Total Ratio - Absolute', 'Undesirable to Desirable Ratio - Absolute']
    # metrics we want
    params = ['Ewma', 'Undesirable to Total Ratio - Absolute']

    plt.rcParams.update({'font.size': 12})

    # for each metric
    for param in params:

        mean_avgs = []
        std_avgs = []
        downscalings = []

        for i in range(len(IDS)):
            # plot convergence and get average metric and reward modifier for each run group
            _, avg, downscaling = plot_avg(IDS[i], param)
            downscalings.append(downscaling)
            mean_avgs.append(np.mean(avg[750:]))
            std_avgs.append(np.std(avg[750:]))
        plt.legend()
        if param == "Ewma":
            param_name = "Ewma Reward"
        else:
            param_name = "Undesirable Behavior Ratio"
        # plt.title('Comparison of ' + param)
        plt.tight_layout()
        plt.show()

        # plot error bar
        plt.errorbar(downscalings, mean_avgs, std_avgs, linestyle='None', marker='x', capsize=3, ecolor='green')
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(downscalings, mean_avgs)

        # use red as color for regression line
        plt.plot(downscalings, slope * np.array(downscalings) + intercept, color='red', label=f'$R^2={r_value:.3f}$')
        plt.xlabel('Reward Modifier', fontsize=16)
        plt.ylabel(param_name + ' (Mean and Std)', fontsize=16)
        plt.legend(fontsize=14)
        print(plt.rcParams['font.size'])

        plt.show()


if __name__ == '__main__':
    """
    Display the graphs shown in the paper.
    """
    while True:
        a = input(
            "Input 1 for the graphs we present in the papers and for the reward convergence of each model group, input 2 for the time ratio between the original and the new models.\n")
        if int(a) == 1:
            # plot the undesirable behavior ratios and average ewma rewards, and also the reward convergence
            # for each reward modifier
            compare_avgs_of_runs()
        if int(a) == 2:
            # We use these runs to compare time
            IDSSSS = [
                ['00', '01', '02', '03', '04'],
                ['10', '11', '12', '13', '14'],
                ['20', '21', '22', '23', '24'],
                ['30', '31', '32', '33', '34'],
                ['40', '41', '42', '43', '44'],
                ['50', '51', '52', '53', '54'],
                ['60', '61', '62', '63', '64'],
                ['70', '71', '72', '73', '74'],
                ['80', '81', '82', '83', '84'],
            ]

            # get time overhead
            get_times_by_models(IDSSSS)
    #run_ids = ['id15' + str(i) + str(j) for i in range(10) for j in range(1)]