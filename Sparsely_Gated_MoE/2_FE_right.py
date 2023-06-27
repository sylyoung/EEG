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
from os import walk

warnings.filterwarnings("ignore")

# Parameters
sample_rate = 256
fs = 256  # sampling rate
Tau = 4
M = 10
R = 0.3
Band = np.arange(1, 128)  # [1, ..., 127] numpy array less than half of the sample frequency
DE = 10

# Read files
zero = next(walk("data/prepro/class0/"), (None, None, []))[2]  # [] if no file
one = next(walk("data/prepro/class1/"), (None, None, []))[2]  # [] if no file

# each channel extracts 74 features, totally 74 * 18 features
for i in range(len(zero)):
    print('test process file here:' + zero[i][15:-4])
    # if zero[i][15:-4] != '50':
    #     continue
    path0 = "data/prepro/class0/" + zero[i]
    path1 = "data/prepro/class1/" + one[i]
    x0 = loadmat(path0)['X']
    x1 = loadmat(path1)['X']

    # Extract features for inter-ictal states
    time_feature_19_0 = np.zeros((x0.shape[0], 2))
    freq_feature_19_0 = np.zeros((x0.shape[0], 2))
    tf_feature_19_0 = np.zeros((x0.shape[0], 2))
    nonlinear_feature_19_0 = np.zeros((x0.shape[0], 2))

    for c in range(x0.shape[2]):  # channel_num
        print(x0.shape)
        print('This is the class0 ' + str(c + 1) + ' channel: ')
        x = x0[:, :, c]
        time_feature = []
        freq_feature = []
        tf_feature = []
        nonlinear_feature = []
        for j in range(x.shape[0]):
            trans = x[j, :]
            trans = np.array([trans])  # (1024, ) → 1 * 1024
            trans = pd.DataFrame(trans, columns=[str(x) for x in range(1024)])  # TODO 1024 can be changed.

            temp1 = timedomain.timedomain(trans)
            matrix1 = np.array(temp1.time_main(mysteps=5))
            time_feature.append(matrix1)  # time domain features    TODO: 数据格式

            temp2 = freqdomain.freqdomain(trans, myfs=fs)
            matrix2 = np.array(temp2.main_freq(percent1=0.5, percent2=0.8, percent3=0.95))
            freq_feature.append(matrix2)  # frequency domain features

            temp3 = timefreq.timefreq(trans, myfs=fs)
            matrix3 = np.array(temp3.main_tf(smoothwindow=100))
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
    time_feature_19_0 = np.delete(time_feature_19_0, [0, 1], axis=1)
    freq_feature_19_0 = np.delete(freq_feature_19_0, [0, 1], axis=1)
    tf_feature_19_0 = np.delete(tf_feature_19_0, [0, 1], axis=1)
    nonlinear_feature_19_0 = np.delete(nonlinear_feature_19_0, [0, 1], axis=1)
    name0 = zero[i][15:-4]
    file1 = 'data/feature_correct/' + name0 + '_inter_feature.npz'
    np.savez(file1, time=time_feature_19_0, freq=freq_feature_19_0, tf=tf_feature_19_0, entropy=nonlinear_feature_19_0)
    print('Subejct_' + name0 + ' inter features saved!')

    # Extract features for ictal states
    time_feature_19_1 = np.zeros((x1.shape[0], 2))
    freq_feature_19_1 = np.zeros((x1.shape[0], 2))
    tf_feature_19_1 = np.zeros((x1.shape[0], 2))
    nonlinear_feature_19_1 = np.zeros((x1.shape[0], 2))

    # Fix fts_labels_r3 format
    for c in range(x1.shape[2]):  # channel_num: 19
        print(x1.shape)
        print('This is the class1 ' + str(c + 1) + ' channel: ')
        x = x1[:, :, c]
        time_feature = []
        freq_feature = []
        tf_feature = []
        nonlinear_feature = []
        for j in range(x.shape[0]):
            trans = x[j, :]
            trans = np.array([trans])  # (1024, ) --→ 1 * 1024
            trans = pd.DataFrame(trans, columns=[str(x) for x in range(1024)])  # numpy 2 pandas dataframe

            temp1 = timedomain.timedomain(trans)
            matrix1 = np.array(temp1.time_main(mysteps=5))
            time_feature.append(matrix1)  # time domain features    TODO: 数据格式

            temp2 = freqdomain.freqdomain(trans, myfs=fs)
            matrix2 = np.array(temp2.main_freq(percent1=0.5, percent2=0.8, percent3=0.95))
            freq_feature.append(matrix2)  # frequency domain features

            temp3 = timefreq.timefreq(trans, myfs=fs)
            matrix3 = np.array(temp3.main_tf(smoothwindow=100))
            tf_feature.append(matrix3)  # time-frequency domain features

            temp4 = nonlinear.nonlinear(trans, myfs=fs)
            matrix4 = np.array(temp4.nonlinear_main(tau=Tau, m=M, r=R, de=DE, n_perm=4, n_lya=40, band=Band))
            nonlinear_feature.append(matrix4)  # nonlinear analysis
        time_feature = np.array(time_feature)
        freq_feature = np.array(freq_feature)
        tf_feature = np.array(tf_feature)
        nonlinear_feature = np.array(nonlinear_feature)
        time_feature_19_1 = np.append(time_feature_19_1, time_feature, axis=1)
        freq_feature_19_1 = np.append(freq_feature_19_1, freq_feature, axis=1)
        tf_feature_19_1 = np.append(tf_feature_19_1, tf_feature, axis=1)
        nonlinear_feature_19_1 = np.append(nonlinear_feature_19_1, nonlinear_feature, axis=1)
    time_feature_19_1 = np.delete(time_feature_19_1, [0, 1], axis=1)
    freq_feature_19_1 = np.delete(freq_feature_19_1, [0, 1], axis=1)
    tf_feature_19_1 = np.delete(tf_feature_19_1, [0, 1], axis=1)
    nonlinear_feature_19_1 = np.delete(nonlinear_feature_19_1, [0, 1], axis=1)
    name1 = one[i][15:-4]
    file2 = 'data/feature_correct/' + name1 + '_ictal_feature.npz'
    np.savez(file2, time=time_feature_19_1, freq=freq_feature_19_1, tf=tf_feature_19_1, entropy=nonlinear_feature_19_1)
    print('Subejct_' + name1 + ' ictal features saved!')
