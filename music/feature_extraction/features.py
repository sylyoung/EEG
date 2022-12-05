# =====================================
# 2021.03.09 zwwang
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

# Parameters # syl: 我不知道这些是干嘛的，没改过
Tau = 4 
M = 10
R = 0.3
DE = 10

# Parameters
fs = 250 # Hz
Band = np.arange(1, fs // 2)

dataset = 'Music'
# Read files
x0 = np.load('./data/' + dataset + '/X.npy')  # (trials, channel, time_samples)
# save path
save_path = './data/' + dataset + '_inter_feature.npz'

ts = x0.shape[2]

x0 = np.transpose(x0, (0, 2, 1))  # (trials, time_samples, channels)

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
    # 16 17 33 8
time_feature_19_0 = np.delete(time_feature_19_0, [0, 1], axis=1)
freq_feature_19_0 = np.delete(freq_feature_19_0, [0, 1], axis=1)
tf_feature_19_0 = np.delete(tf_feature_19_0, [0, 1], axis=1)
nonlinear_feature_19_0 = np.delete(nonlinear_feature_19_0, [0, 1], axis=1)

np.savez(save_path, time=time_feature_19_0, freq=freq_feature_19_0, tf=tf_feature_19_0, entropy=nonlinear_feature_19_0)
print(dataset + ' inter features saved!')
