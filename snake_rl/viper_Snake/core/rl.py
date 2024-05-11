# Copyright 2017-2018 MIT
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import pickle

import numpy as np
from ..util.log import *


# def get_rollouts(env, policy, render, n_batch_rollouts):
#     rollouts = []
#     for i in range(n_batch_rollouts):
#         rollouts.extend(get_rollout(env, policy, render))
#     return rollouts


def _sample(obss, acts, qs, max_pts, is_reweight):
    # Step 1: Compute probabilities
    ps = np.max(qs, axis=1) - np.min(qs, axis=1)
    ps = ps / np.sum(ps)

    # Step 2: Sample points
    if is_reweight:
        # According to p(s)
        idx = np.random.choice(len(obss), size=min(max_pts, np.sum(ps > 0)), p=ps)
    else:
        # Uniformly (without replacement)
        idx = np.random.choice(len(obss), size=min(max_pts, np.sum(ps > 0)), replace=False)

        # Step 3: Obtain sampled indices
    return obss[idx], acts[idx], qs[idx]


def test_policy(student, n_test_rollouts, it):
    cum_rew = 0.0
    for i in range(n_test_rollouts):
        student_trace = student.get_rollout_student(i)
        cum_rew += sum((rew for _, _, rew in student_trace))
    return cum_rew / n_test_rollouts

def transform(state):
    return state

def identify_best_policy(students, n_test_rollouts):
    log('Initial policy count: {}'.format(len(students)), INFO)
    it = 0
    # cut policies by half on each iteration
    while len(students) > 1:
        # Step 1: Sort policies by current estimated reward
        students = sorted(students, key=lambda entry: -entry[1])

        # Step 2: Prune second half of policies
        n_students = int((len(students) + 1) / 2)
        log('Current policy count: {}'.format(n_students), INFO)

        # Step 3: build new policies
        new_policies = []
        for i in range(n_students):
            policy, rew = students[i]
            new_rew = test_policy(policy, n_test_rollouts, it)
            new_policies.append((policy, new_rew))
            log('Reward update: {} -> {}'.format(rew, new_rew), INFO)

        students = new_policies
        it += 1

    if len(students) != 1:
        raise Exception()

    return students[0][0]


# def _get_action_sequences_helper(trace, seq_len):
#     acts = [act for _, act, _ in trace]
#     seqs = []
#     for i in range(len(acts) - seq_len + 1):
#         seqs.append(acts[i:i + seq_len])
#     return seqs
#
#
# def get_action_sequences(env, policy, seq_len, n_rollouts):
#     # Step 1: Get action sequences
#     seqs = []
#     for _ in range(n_rollouts):
#         trace = get_rollout(env, policy, False)
#         seqs.extend(_get_action_sequences_helper(trace, seq_len))
#
#     # Step 2: Bin action sequences
#     counter = {}
#     for seq in seqs:
#         s = str(seq)
#         if s in counter:
#             counter[s] += 1
#         else:
#             counter[s] = 1
#
#     # Step 3: Sort action sequences
#     seqs_sorted = sorted(list(counter.items()), key=lambda pair: -pair[1])
#
#     return seqs_sorted


def train_dagger(teacher, student, max_iters, n_batch_rollouts, max_samples, train_frac,
                 is_reweight, n_test_rollouts):
    # Step 0: Setup
    obss, acts, qs = [], [], []
    students = []

    #####################
    ### START ADDITION
    #####################
    # if we saved previously
    try:
        with open('obss_reduced.pkl', 'rb') as f:
            obss = pickle.load(f)
        with open('acts_reduced.pkl', 'rb') as f:
            acts = pickle.load(f)
        with open('qs_reduced.pkl', 'rb') as f:
            qs = pickle.load(f)
    #####################
    ### END ADDITION
    #####################
    except:
        # Step 1: Generate some supervised traces into the buffer
        trace = teacher.get_rollouts(n_batch_rollouts) # nbatch * max_steps
        obss.extend((obs for obs, _, _ in trace))
        acts.extend((act for _, act, _ in trace))
        qs.extend(teacher.predict_q(np.array([obs for obs, _, _ in trace])))

        cur_rew = sum((rew for _, _, rew in trace)) / n_batch_rollouts
        log('Teacher reward: {}'.format(cur_rew), INFO)

        #####################
        ### START ADDITION
        #####################
        # pickling
        with open('obss_reduced.pkl', 'wb+') as f:
            pickle.dump(obss, f)

        with open('acts_reduced.pkl', 'wb+') as f:
            pickle.dump(acts, f)

        with open('qs_reduced.pkl', 'wb+') as f:
            pickle.dump(qs, f)

    try:
        with open('students_reduced.pkl', 'rb') as f:
            students = pickle.load(f)
    except:
        x = 1
        #####################
        ### END ADDITION
        #####################

    # Step 2: Dagger outer loop
    for i in range(len(students), max_iters):
        log('Iteration {}/{}'.format(i, max_iters), INFO)

        # Step 2a: Train from a random subset of aggregated data
        cur_obss, cur_acts, cur_qs = _sample(np.array(obss), np.array(acts), np.array(qs), max_samples, is_reweight)
        log('Training student with {} points'.format(len(cur_obss)), INFO)
        student.train([obsss[0].tolist() for obsss in cur_obss], cur_acts.tolist(), train_frac)

        # Step 2b: Generate trace using student
        student_trace = student.get_rollouts(n_batch_rollouts, i) # nbatch * max_steps * max_iters
        student_obss = [obs for obs, _, _ in student_trace]

        # Step 2c: Query the oracle for supervision
        teacher_qs = teacher.predict_q(
            np.array(student_obss))  # at the interface level, order matters, since teacher.predict may run updates
        teacher_acts = teacher.predict(student_obss)

        # Step 2d: Add the augmented state-action pairs back to aggregate
        obss.extend((obs for obs in student_obss))
        acts.extend(teacher_acts)
        qs.extend(teacher_qs)

        # Step 2e: Estimate the reward
        cur_rew = sum((rew for _, _, rew in student_trace)) / n_batch_rollouts
        log('Student reward: {}'.format(cur_rew), INFO)

        students.append((student.clone(), cur_rew))

        #####################
        ### START ADDITION
        #####################
        # pickle when possible
        try:

            for stud, r in students:
                stud.pre_save()

            with open('students_reduced.pkl', 'wb+') as f:
                pickle.dump(students, f)

            for stud, r in students:
                stud.post_save()
        except:
            x = 1
        #####################
        ### END ADDITION
        #####################

    max_student = identify_best_policy(students, n_test_rollouts)

    return max_student