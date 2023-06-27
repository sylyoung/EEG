# =====================================
# 2021.03.09 vivi
# extract 4 class features: time, frequency, time-freq, and entropy.
# =====================================
import mne
import numpy as np
import pandas as pd
from utils import timedomain, freqdomain, timefreq, nonlinear
from scipy.io import savemat, loadmat
import itertools as it
import warnings

from scipy.linalg import fractional_matrix_power
from os import walk

warnings.filterwarnings("ignore")


def EA(x):
    """
    Parameters
    ----------
    x : numpy array
        data of shape (num_samples, num_channels, num_time_samples)

    Returns
    ----------
    XEA : numpy array
        data of shape (num_samples, num_channels, num_time_samples)
    """
    cov = np.zeros((x.shape[0], x.shape[1], x.shape[1]))
    for i in range(x.shape[0]):
        cov[i] = np.cov(x[i])
    refEA = np.mean(cov, 0)
    sqrtRefEA = fractional_matrix_power(refEA, -0.5)
    XEA = np.zeros(x.shape)
    for i in range(x.shape[0]):
        XEA[i] = np.dot(sqrtRefEA, x[i])
    return XEA


def data_alignment(X, num_subjects):
    '''
    :param X: np array, EEG data
    :param num_subjects: int, number of total subjects in X
    :return: np array, aligned EEG data
    '''
    # subject-wise EA
    print('before EA:', X.shape)
    out = []
    for i in range(num_subjects):
        tmp_x = EA(X[X.shape[0] // num_subjects * i:X.shape[0] // num_subjects * (i + 1), :, :])
        out.append(tmp_x)
    X = np.concatenate(out, axis=0)
    print('after EA:', X.shape)
    return X

def data_alignment_two_sessions(X, num_subjects, session_split):
    '''
    :param X: np array, EEG data of two sessions
    :param num_subjects: int, number of total subjects in X
    :param session_split: float, session split ratio
    :return: np array, aligned EEG data
    '''
    # subject-wise EA
    print('before EA:', X.shape)
    out = []
    for i in range(num_subjects):
        subj_trial_num = int(X.shape[0] // num_subjects)
        first_session_num = int(subj_trial_num * session_split)
        print('first_session_num, subj_trial_num:', first_session_num, subj_trial_num)
        tmp_x = EA(X[subj_trial_num * i:subj_trial_num * i + first_session_num, :, :])
        out.append(tmp_x)
        tmp_x = EA(X[subj_trial_num * i + first_session_num:subj_trial_num * (i + 1), :, :])
        out.append(tmp_x)
        print(tmp_x.shape)
    X = np.concatenate(out, axis=0)
    print('after EA:', X.shape)
    return X


# since some subjects contain 3 sessions in BNCI2015001
def data_alignment_two_sessions_BNCI2015001(X, num_subjects, session_split):
    '''
    :param X: np array, EEG data of two sessions
    :param num_subjects: int, number of total subjects in X
    :param session_split: float, session split ratio
    :return: np array, aligned EEG data
    '''
    # subject-wise EA
    print('before EA:', X.shape)

    subj_trial_num_arr = None
    first_session_num_decided = None
    if dataset == 'BNCI2015001':
        subj_trial_num_arr = [400, 400, 400, 400, 400, 400, 400, 600, 600, 600, 600, 400]
        first_session_num_decided = 200

    out = []
    for i in range(num_subjects):
        if subj_trial_num_arr is not None:
            subj_trial_num = subj_trial_num_arr[i]
            total_sum = int(np.sum(subj_trial_num_arr[:i]))
        else:
            subj_trial_num = int(X.shape[0] // num_subjects)
        if first_session_num_decided is not None:
            first_session_num = first_session_num_decided
        else:
            first_session_num = int(subj_trial_num * session_split)


        print('first_session_num, subj_trial_num:', first_session_num, subj_trial_num)
        # print(first_session_num, subj_trial_num)
        #tmp_x = EA(X[subj_trial_num * i:subj_trial_num * i + first_session_num, :, :])
        tmp_x = EA(X[total_sum:total_sum + first_session_num, :, :])
        out.append(tmp_x)
        print(tmp_x.shape)
        #tmp_x = EA(X[subj_trial_num * i + first_session_num:subj_trial_num * (i + 1), :, :])
        #tmp_x = EA(X[subj_trial_num * i + first_session_num:subj_trial_num * i + first_session_num * 2, :, :])
        tmp_x = EA(X[total_sum + first_session_num:total_sum + first_session_num * 2, :, :])
        out.append(tmp_x)
        print(tmp_x.shape)
        if subj_trial_num // first_session_num_decided == 3:
            #tmp_x = EA(X[subj_trial_num * i + first_session_num * 2:subj_trial_num * (i + 1), :, :])
            tmp_x = EA(X[total_sum + first_session_num * 2:total_sum + subj_trial_num, :, :])
            out.append(tmp_x)
            print(tmp_x.shape)

    X = np.concatenate(out, axis=0)
    print('after EA:', X.shape)
    return X


def data_alignment_ERP(X, num_subjects):
    '''
    :param X: np array, EEG data of two sessions
    :param num_subjects: int, number of total subjects in X
    :param session_split: float, session split ratio
    :return: np array, aligned EEG data
    '''
    # subject-wise EA
    print('before EA:', X.shape)
    out = []
    for i in range(num_subjects):
        subj_trial_num = int(X.shape[0] // num_subjects)
        tmp_x = EA(X[subj_trial_num * i:subj_trial_num * (i + 1), :, :])
        out.append(tmp_x)
        print(tmp_x.shape)
    X = np.concatenate(out, axis=0)
    print('after EA:', X.shape)
    return X


def time_cut(data, cut_percentage):
    # Time Cutting: cut at a certain percentage of the time. data: (..., ..., time_samples)
    data = data[:, :, :int(data.shape[2] * cut_percentage)]
    return data


# Parameters
num_subjects = 1
fs = 250  # 250, 512, 800
Tau = 4
M = 10
R = 0.3
Band = np.arange(1, fs // 2)  # [1, ..., 127] numpy array less than half of the sample frequency
DE = 10

#dataset = 'SEED-V'
#dataset = 'BNCI2014008'
dataset = 'Music'

#dataset_arr = ['BNCI2014001', 'BNCI2014002', 'MI1', 'BNCI2015001']

# Read files
x0 = np.load('./data/' + dataset + '/X10.npy')
#x0 = time_cut(x0, cut_percentage=0.8) # BNCI2014008

#x0 = data_alignment_two_sessions(x0, 8, 0.5)
#x0 = data_alignment_ERP(x0, num_subjects)

#x0 = np.load('./data/' + dataset + '/X_4sec.npy')
ts = x0.shape[2]

x0 = np.transpose(x0, (0, 2, 1)) # (trials, channel, time_samples)

#zero = next(walk("data/prepro/class0/"), (None, None, []))[2]  # [] if no file
#one = next(walk("data/prepro/class1/"), (None, None, []))[2]  # [] if no file

# each channel extracts 74 features, totally 74 * 18 features
#for i in range(len(x0)):

# Extract features for inter-ictal states
time_feature_19_0 = np.zeros((x0.shape[0], 2))
freq_feature_19_0 = np.zeros((x0.shape[0], 2))
tf_feature_19_0 = np.zeros((x0.shape[0], 2))
nonlinear_feature_19_0 = np.zeros((x0.shape[0], 2))

for c in range(x0.shape[2]):  # channel num
    print(x0.shape)
    print('channel: ', str(c))
    x = x0[:, :, c]
    time_feature = []
    freq_feature = []
    tf_feature = []
    nonlinear_feature = []
    for j in range(x.shape[0]): # sample num
        trans = x[j, :]

        trans = np.array([trans])

        trans = pd.DataFrame(trans, columns=[str(x) for x in range(ts)])

        temp1 = timedomain.timedomain(trans)
        matrix1 = np.array(temp1.time_main(mysteps=5))
        time_feature.append(matrix1)  # time domain features

        temp2 = freqdomain.freqdomain(trans, myfs=fs)
        matrix2 = np.array(temp2.main_freq(percent1=0.5, percent2=0.8, percent3=0.95))
        freq_feature.append(matrix2)  # frequency domain features

        temp3 = timefreq.timefreq(trans, myfs=fs)
        matrix3 = np.array(temp3.main_tf(smoothwindow=100))  # 100 ; decrease?
        tf_feature.append(matrix3)  # time-frequency domain features

        temp4 = nonlinear.nonlinear(trans, myfs=fs)
        matrix4 = np.array(temp4.nonlinear_main(tau=Tau, m=M, r=R, de=DE, n_perm=4, n_lya=40, band=Band))
        nonlinear_feature.append(matrix4)  # nonlinear analysis
    time_feature = np.array(time_feature)
    freq_feature = np.array(freq_feature)
    tf_feature = np.array(tf_feature)
    nonlinear_feature = np.array(nonlinear_feature)

    time_feature_19_0 = np.append(time_feature_19_0, time_feature, axis=1)
    freq_feature_19_0 = np.append(freq_feature_19_0, freq_feature, axis=1)
    tf_feature_19_0 = np.append(tf_feature_19_0, tf_feature, axis=1)
    nonlinear_feature_19_0 = np.append(nonlinear_feature_19_0, nonlinear_feature, axis=1)
    # 16 17 33 8
time_feature_19_0 = np.delete(time_feature_19_0, [0, 1], axis=1)
freq_feature_19_0 = np.delete(freq_feature_19_0, [0, 1], axis=1)
tf_feature_19_0 = np.delete(tf_feature_19_0, [0, 1], axis=1)
nonlinear_feature_19_0 = np.delete(nonlinear_feature_19_0, [0, 1], axis=1)
file = './data/' + dataset + '_inter_feature_EA\'d.npz'
np.savez(file, time=time_feature_19_0, freq=freq_feature_19_0, tf=tf_feature_19_0, entropy=nonlinear_feature_19_0)
print(dataset + ' inter features saved!')
