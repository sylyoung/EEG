import sys
import os

import mne
import scipy
import numpy as np
import scipy.io
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from mne.decoding import CSP
from mne.preprocessing import Xdawn
from mne.time_frequency import tfr_morlet, psd_multitaper, psd_welch
from sklearn.manifold import TSNE

from moabb.datasets import BNCI2014001, BNCI2014002, BNCI2014008, BNCI2014009, BNCI2015003, BNCI2015004, EPFLP300, BNCI2014004, BNCI2015001
from moabb.paradigms import MotorImagery, P300

lib_path = os.path.abspath(os.path.join('..'))
sys.path.append(lib_path)

from EEG.ml.main import data_loader, data_alignment
from EEG.utils.data_utils import dataset_to_file


def plot_eeg(config):

    ########################################
    mat_data = scipy.io.loadmat(config['path'])
    # one trial
    data = mat_data[config['x_name']]
    # sampling rate
    sampling_rate = config['sampling_rate']
    # channel names
    ch_names = config['ch_names']
    # assume all EEG channels
    ch_types = ['eeg'] * len(ch_names)
    # create info
    info = mne.create_info(ch_names=ch_names, ch_types=ch_types, sfreq=sampling_rate)
    # TODO triggers error
    # set 62-channel electrode cap according to the international 10-20 system
    #info.set_montage('standard_1020')
    # description
    info['description'] = config['description']
    ########################################

    print('#' * 20 + 'info str start' + '#' * 20)
    print(info)
    print('#' * 20 + 'info str end' + '#' * 20)
    print('data.shape:', data.shape)
    #print(data[:5, :5])

    assert len(data.shape) == 2, 'Wrong data dimension! Please do not include more than 1 trial in data!'
    if data.shape[0] > 100:
        print('Wrong data shape! Transposing data...')
        data = np.transpose(data)
        # shape: (n_channels, n_times)
        print('data.shape:', data.shape)

    raw_arr = mne.io.RawArray(data, info)
    print(raw_arr)

    # 20uF scaling https://blog.csdn.net/qq_37813206/article/details/116428688
    # bandpass filter between 0.3 to 50 Hz https://ieeexplore.ieee.org/abstract/document/7104132 Investigating Critical Frequency Bands and Channels for EEG-Based Emotion Recognition with Deep Neural Networks
    raw_arr.plot(block=True, title=config['description'], scalings=dict(eeg=20), highpass=0.3, lowpass=50, show=True, n_channels=len(ch_names))
    #plt.savefig('./results/figures/plot.png')


    '''
    processed_data = raw_arr[:][0]
    print(processed_data.shape)
    print(processed_data[:5, :5])
    p = (data == processed_data)
    cnt = 0
    for i in range(len(p)):
        for j in range(len(p[0])):
            if p[i][j] != True:
                cnt += 1
    print(cnt, data.shape[0] * data.shape[1])
    #np.save('./data_mod/BNCI2014-001/' + 's0_bandpass.mat', processed_data)
    np.save('./data_mod/SEED/' + 's0_bandpass', processed_data)
    '''


def plot_mne_eeg():
    dataset = BNCI2015001()
    paradigm = MotorImagery(n_classes=2)
    print('preparing data...')
    X, labels, meta = paradigm.get_data(dataset=dataset, subjects=dataset.subject_list[:])
    ar_unique, cnts = np.unique(labels, return_counts=True)
    print("labels:", ar_unique)
    print("Counts:", cnts)

    X_with_info, _, _ = paradigm.get_data(dataset=dataset, subjects=dataset.subject_list[:], return_epochs=True)
    raw = mne.io.RawArray(X[0], X_with_info.info)
    raw.plot(duration=5, n_channels=22, block=True, scalings=dict(eeg=20), highpass=0.3, lowpass=50, show=True)
    plt.savefig('./results/figures/plot.png')


def plot_ensemble_results():
    data_name_arr = ['BNCI2014001', 'BNCI2014002', 'BNCI2015001']

    # rerun
    arr1 = [[77.92, 78.6, 78.91, 79.26, 79.28, 79.42, 79.41, 79.47, 79.46],
     [76.55, 78.48, 78.16, 79.14, 79.05, 79.32, 79.24, 79.42, 79.44],
     [76.76, 78.08, 78.53, 79.01, 79.2, 79.17, 79.57, 79.19, 79.68],
     [73.77, 78.55, 79.0, 79.26, 79.33, 79.49, 79.55, 79.51, 79.52]]
    arr2 = [[79.49, 79.56, 79.69, 79.8, 79.83, 79.68, 79.79, 79.76, 79.76],
     [78.64, 79.51, 79.57, 79.81, 79.54, 79.66, 79.65, 79.61, 79.59],
     [78.56, 79.24, 79.39, 79.74, 79.67, 79.83, 79.66, 79.72, 79.61],
     [76.72, 79.56, 79.6, 79.91, 79.85, 79.75, 79.79, 79.72, 79.77]]
    arr3 = [[77.6, 77.84, 78.07, 78.25, 78.29, 78.42, 78.38, 78.57, 78.5],
     [77.13, 77.7, 77.77, 77.99, 78.11, 78.24, 78.21, 78.41, 78.41],
     [77.06, 77.38, 77.4, 78.01, 77.8, 78.25, 78.15, 78.42, 78.36],
     [75.51, 77.78, 77.99, 78.16, 78.22, 78.38, 78.35, 78.5, 78.45]]
    arrs = [arr1, arr2, arr3]


    '''
    # standard ensemble results
    arr1 = [[77.84, 78.64, 78.96, 79.17, 79.4, 79.42, 79.39, 79.45, 79.47], [76.55, 78.48, 78.16, 79.14, 79.05, 79.32, 79.24, 79.42, 79.44], [76.5, 78.17, 78.63, 78.96, 79.22, 79.11, 79.66, 79.2, 79.72], [73.93, 78.56, 78.98, 79.3, 79.32, 79.42, 79.55, 79.54, 79.54]]
    arr2 = [[79.37, 79.54, 79.6, 79.69, 79.8, 79.63, 79.76, 79.76, 79.71], [78.69, 79.42, 79.56, 79.71, 79.99, 79.62, 79.86, 79.61, 79.75], [78.58, 79.32, 79.3, 79.69, 79.64, 79.78, 79.6, 79.67, 79.54], [76.66, 79.56, 79.54, 79.81, 79.82, 79.72, 79.76, 79.69, 79.66]]
    arr3 = [[77.69, 77.83, 78.02, 78.26, 78.25, 78.42, 78.36, 78.53, 78.5], [77.46, 77.69, 78.02, 77.98, 78.21, 78.22, 78.26, 78.34, 78.32], [77.25, 77.43, 77.37, 78.02, 77.8, 78.22, 78.17, 78.38, 78.38], [75.81, 77.77, 77.92, 78.2, 78.18, 78.39, 78.32, 78.46, 78.46]]
    arrs = [arr1, arr2, arr3]
    '''
    '''
    # second half ensemble results
    arr1 = [[76.94, 77.05, 77.73, 77.55, 77.82, 77.75, 77.84, 77.96, 77.72], [75.63, 77.16, 77.5, 77.62, 77.73, 77.78, 77.61, 77.78, 77.81], [75.37, 77.04, 77.79, 77.47, 77.92, 77.81, 77.85, 77.89, 78.00], [73.97, 76.99, 77.9, 77.73, 77.89, 78.12, 78.19, 78.23, 78.06]]
    arr2 = [[80.36, 80.23, 80.36, 80.31, 80.34, 80.3, 80.41, 80.27, 80.43], [80.21, 80.34, 80.71, 80.49, 81.04, 80.46, 80.77, 80.29, 80.61], [79.96, 80.24, 80.21, 80.32, 80.23, 80.36, 80.31, 80.29, 80.41], [79.53, 80.19, 80.3, 80.33, 80.41, 80.31, 80.47, 80.21, 80.44]]
    arr3 = [[79.03, 78.99, 79.19, 79.46, 79.41, 79.61, 79.42, 79.59, 79.39], [78.95, 78.95, 79.43, 79.18, 79.64, 79.42, 79.71, 79.55, 79.75], [78.76, 78.94, 78.79, 78.88, 78.96, 79.07, 79.15, 79.55, 79.45], [78.14, 78.98, 79.13, 79.44, 79.29, 79.53, 79.37, 79.52, 79.41]]
    arrs = [arr1, arr2, arr3]
    '''

    '''
    # standard with 6
    arr1 = [[77.92, 78.6, 78.91, 79.26, 79.38, 79.42, 79.41, 79.47, 79.46], [76.9, 78.48, 77.99, 79.14, 78.94, 79.32, 79.31, 79.42, 79.25], [76.84, 78.17, 78.64, 79.07, 79.17, 79.12, 79.61, 79.24, 79.74], [73.8, 78.54, 79.02, 79.3, 79.33, 79.45, 79.57, 79.52, 79.52], [76.86, 78.49, 78.46, 79.14, 79.01, 79.32, 79.54, 79.42, 79.77], [74.33, 78.58, 78.97, 79.17, 79.31, 79.44, 79.43, 79.51, 79.47]]
    arr2 = [[79.49, 79.56, 79.69, 79.8, 79.83, 79.68, 79.79, 79.76, 79.76], [78.57, 79.51, 79.37, 79.81, 79.66, 79.66, 79.49, 79.61, 79.72], [78.56, 79.39, 79.41, 79.81, 79.66, 79.84, 79.64, 79.7, 79.56], [76.82, 79.57, 79.61, 79.91, 79.85, 79.74, 79.79, 79.72, 79.74], [78.54, 79.5, 79.63, 79.81, 79.7, 79.66, 79.66, 79.61, 79.64], [75.44, 79.56, 79.71, 79.81, 79.86, 79.69, 79.82, 79.75, 79.76]]
    arr3 = [[77.6, 77.84, 78.07, 78.25, 78.29, 78.42, 78.38, 78.57, 78.5], [77.0, 77.7, 77.88, 77.99, 78.07, 78.24, 78.18, 78.41, 78.23], [77.08, 77.44, 77.42, 78.03, 77.83, 78.25, 78.18, 78.45, 78.33], [75.52, 77.79, 77.98, 78.17, 78.23, 78.38, 78.35, 78.5, 78.45], [76.99, 77.7, 77.71, 77.99, 77.89, 78.24, 78.13, 78.41, 78.29], [76.0, 77.8, 78.08, 78.22, 78.26, 78.42, 78.36, 78.56, 78.5]]
    arrs = [arr1, arr2, arr3]
    '''

    '''
    # second half with 6
    arr1 = [[77.15, 77.13, 77.62, 77.72, 77.89, 77.78, 77.84, 77.98, 77.76], [75.71, 77.16, 77.27, 77.62, 77.59, 77.78, 77.65, 77.78, 77.55], [75.65, 74.15, 76.33, 77.82, 77.92, 78.15, 77.92, 77.5, 78.52], [74.12, 77.15, 78.02, 78.33, 77.79, 78.35, 78.21, 78.56, 78.32], [75.93, 77.18, 77.38, 77.62, 77.95, 77.78, 78.36, 77.78, 78.19], [74.81, 77.11, 77.69, 77.69, 77.84, 77.81, 77.85, 78.02, 77.76]]
    arr2 = [[80.39, 80.2, 80.39, 80.37, 80.31, 80.24, 80.37, 80.27, 80.44], [79.9, 80.36, 80.61, 80.54, 80.63, 80.46, 80.4, 80.31, 80.3], [79.91, 75.53, 77.63, 80.64, 80.21, 80.49, 79.49, 78.67, 80.44], [78.69, 80.01, 80.3, 80.33, 80.34, 80.23, 80.34, 80.29, 80.5], [79.94, 80.36, 80.8, 80.54, 80.8, 80.46, 80.7, 80.31, 80.64], [78.24, 80.29, 80.4, 80.4, 80.41, 80.24, 80.41, 80.3, 80.46]]
    arr3 = [[78.83, 78.9, 79.16, 79.42, 79.42, 79.58, 79.45, 79.62, 79.43], [78.55, 78.85, 78.88, 79.13, 79.22, 79.4, 79.44, 79.6, 79.51], [78.68, 74.42, 76.93, 79.22, 79.03, 79.47, 77.43, 77.07, 79.6], [77.37, 78.9, 79.14, 79.36, 79.3, 79.47, 79.39, 79.63, 79.48], [78.61, 78.85, 79.09, 79.13, 79.23, 79.4, 79.28, 79.6, 79.46], [78.51, 78.93, 79.18, 79.38, 79.33, 79.58, 79.42, 79.59, 79.41]]
    arrs = [arr1, arr2, arr3]
    '''

    '''
    # remove bad
    arr1 = [[86.72, 87.21, 87.47, 87.68, 87.75, 87.89, 88.01, 88.01, 87.92],
     [85.4, 87.06, 87.08, 87.58, 87.57, 87.89, 87.72, 88.03, 87.97],
     [85.62, 86.56, 86.9, 87.5, 87.62, 87.63, 87.92, 87.75, 88.04],
     [82.28, 87.24, 87.47, 87.64, 87.61, 87.83, 88.0, 87.89, 87.9]]
    arr2 = [[85.2, 85.3, 85.31, 85.42, 85.36, 85.35, 85.41, 85.31, 85.22],
     [84.59, 85.29, 85.24, 85.52, 85.42, 85.43, 85.53, 85.35, 85.41],
     [84.55, 84.94, 85.22, 85.43, 85.52, 85.5, 85.58, 85.3, 85.45],
     [82.1, 85.33, 85.4, 85.52, 85.43, 85.43, 85.43, 85.27, 85.35]]
    arr3 = [[85.46, 85.57, 85.72, 85.82, 85.83, 85.96, 85.92, 86.04, 85.99],
     [84.93, 85.48, 85.46, 85.68, 85.79, 85.92, 85.82, 86.07, 85.98],
     [84.79, 85.06, 85.27, 85.71, 85.62, 85.91, 85.91, 85.99, 86.08],
     [82.72, 85.56, 85.67, 85.83, 85.84, 85.96, 85.91, 86.02, 85.96]]
    arrs = [arr1, arr2, arr3]
    '''
    '''
    # EEGNet + ShallowCNN
    arr1 = [[77.84, 78.58, 79.32, 79.36, 79.68, 79.55, 79.77, 79.68, 79.98], [76.08, 78.47, 78.13, 79.32, 78.8, 79.6, 79.65, 79.71, 79.56], [76.06, 78.02, 78.36, 79.3, 79.42, 79.52, 79.66, 79.85, 79.84], [74.08, 78.57, 79.3, 79.38, 79.68, 79.67, 79.71, 79.71, 79.71]]
    arr2 = [[77.63, 78.31, 78.81, 79.27, 79.14, 79.54, 79.35, 79.74, 79.53], [75.36, 78.29, 78.15, 79.26, 78.92, 79.69, 79.36, 79.96, 79.94], [75.21, 77.76, 77.85, 79.16, 78.94, 79.42, 79.56, 79.39, 79.7], [74.11, 78.06, 78.64, 79.37, 78.96, 79.34, 79.23, 79.42, 79.37]]
    arr3 = [[77.63, 77.84, 78.63, 78.52, 78.82, 78.94, 79.13, 79.19, 79.19], [75.78, 77.74, 77.53, 78.25, 78.55, 78.63, 78.88, 78.9, 78.91], [75.43, 77.53, 77.49, 78.23, 78.08, 78.57, 78.55, 78.76, 78.87], [74.45, 77.74, 78.32, 78.44, 78.59, 78.74, 78.92, 79.02, 79.01]]
    arrs = [arr1, arr2, arr3]
    '''
    '''
    # EEGNet + ShallowCNN remove bad
    arr1 = [[86.33, 87.01, 87.44, 87.69, 87.83, 87.76, 87.89, 87.82, 88.14],
     [85.17, 86.93, 87.1, 87.53, 87.51, 87.62, 87.63, 87.89, 87.85],
     [84.68, 86.31, 86.5, 87.54, 87.5, 87.57, 87.85, 87.89, 88.03],
     [82.43, 86.93, 87.33, 87.6, 87.86, 87.69, 87.96, 87.86, 88.03]]
    arr2 = [[83.01, 83.87, 84.03, 84.59, 84.46, 84.86, 84.75, 85.0, 84.89],
     [80.98, 83.96, 83.75, 84.83, 84.61, 85.17, 85.11, 85.44, 85.33],
     [80.72, 83.33, 83.34, 84.67, 84.46, 85.01, 85.12, 85.13, 85.33],
     [79.35, 83.71, 83.93, 84.64, 84.56, 84.92, 84.84, 84.99, 84.98]]
    arr3 = [[85.06, 85.14, 85.78, 85.86, 85.96, 86.02, 86.12, 86.27, 86.26],
     [83.09, 85.06, 84.98, 85.62, 85.6, 85.9, 85.84, 86.19, 86.11],
     [82.84, 84.78, 85.17, 85.61, 85.67, 85.84, 85.94, 86.15, 86.22],
     [81.29, 85.16, 85.72, 85.84, 85.94, 86.0, 86.08, 86.22, 86.29]]
    arrs = [arr1, arr2, arr3]
    '''
    """
    # np.cov
    arr1 = [[77.92, 78.6, 78.91, 79.26, 79.38, 79.42, 79.41, 79.47, 79.46], [76.73, 78.48, 78.5, 79.14, 78.81, 79.32, 79.21, 79.42, 79.21], [76.67, 77.93, 78.52, 79.0, 79.3, 79.15, 79.49, 79.33, 79.73], [73.07, 78.36, 78.97, 79.21, 79.37, 79.44, 79.49, 79.51, 79.72]]
    arr2 = [[79.49, 79.56, 79.69, 79.8, 79.83, 79.68, 79.79, 79.76, 79.76], [78.51, 79.51, 79.48, 79.81, 79.62, 79.66, 79.56, 79.61, 79.61], [78.36, 79.46, 79.74, 79.84, 80.04, 79.79, 80.0, 79.74, 79.72], [75.84, 79.49, 79.68, 79.85, 79.94, 79.92, 79.94, 79.81, 79.84]]
    arr3 = [[77.6, 77.84, 78.07, 78.25, 78.29, 78.42, 78.38, 78.57, 78.5], [77.14, 77.7, 77.69, 77.99, 77.94, 78.24, 78.26, 78.41, 78.42], [76.89, 77.57, 77.95, 78.1, 78.17, 78.3, 78.24, 78.47, 78.34], [74.92, 77.75, 78.1, 78.21, 78.24, 78.37, 78.38, 78.46, 78.49]]
    arrs = [arr1, arr2, arr3]
    """

    #arr1 = [[55.7, 56.31, 56.5, 56.66, 56.69, 56.7, 56.7, 56.48, 56.52], [55.49, 56.26, 56.76, 56.52, 56.96, 56.5, 56.9, 56.51, 56.83], [50.05, 56.52, 56.69, 56.99, 56.85, 56.97, 56.91, 56.83, 56.87]]
    #arrs = [arr1]

    '''
    T-TIME results for 11 seeds averages on CPU, not 5 seeds on GPU
    0, BNCI2014001, 76.83782, 1.01346, 83.39646, 60.85859, 95.89646, 67.29798, 58.96465, 73.80051, 68.18182, 91.28788, 91.85606,, , , ,
    1, BNCI2014002, 78.9026, 0.59237, 71.81818, 82.81818, 94.45455, 91.63636, 85.18182, 75.45455, 89.18182, 68.81818, 87.90909, 77.90909, 83.54545, 80.54545, 60.72727, 54.63636
    2, BNCI2015001, 77.32955, 0.84527, 97.63636, 94.81818, 94.86364, 88.95455, 88.54545, 67.36364, 81.22727, 66.54545, 64.90909, 66.86364, 59.68182, 56.54545,,
    
    results_without_ensemble = [76.83782, 78.9026, 77.32955]
    '''

    '''
    T-TIME results for 11 seeds averages (second-half) on CPU, not 5 seeds on GPU
    BNCI2014001 77.006
    BNCI2014002 79.999
    BNCI2015001 78.750
    '''
    #results_without_ensemble = [77.006, 79.999, 78.750]

    labels = ["averaging", "voting", "SML-hard", "SML-soft"]
    #labels = ["averaging", "voting", "SML-hard", "SML-soft", "SML-hard w/ training-set", "SML-soft w/ training-set"]
    for i in range(len(arrs)):
        arr = arrs[i]
        arr = np.array(arr)[:, 1:]

        y1 = arr[0]
        y2 = arr[1]
        y3 = arr[2]
        y4 = arr[3]
        #y4 = arr[3]
        #y5 = arr[4]
        #y6 = arr[5]

        #for y in [y1, y2, y3, y4]:
        #    y = y.insert(0, results_without_ensemble[i])

        #x = np.arange(9).astype(int) + 2
        x = np.arange(8).astype(int) + 3

        plt.plot(x, y1, color='brown', marker='x', markersize=4, alpha=0.7, lw=2, label='Averaging')
        plt.plot(x, y2, color='green', marker='o', markersize=4, alpha=0.7, lw=2, label='Voting')
        plt.plot(x, y3, color='steelblue', marker='v', markersize=4, alpha=0.7, lw=2, label='SML-hard')
        plt.plot(x, y4, color='red', marker='^', markersize=4, alpha=0.7, lw=2, label='SML-soft')
        #plt.plot(x, y5, color='yellow', marker='>', markersize=4, alpha=0.7, lw=2, label='SML-hard w/ training-set')
        #plt.plot(x, y6, color='black', marker='<', markersize=4, alpha=0.7, lw=2, label='SML-soft w/ training-set')

        plt.grid(axis='y', color='grey', linestyle='--', lw=1, alpha=0.5)
        plt.legend(loc="lower right", fontsize=13)
        plt.xlabel("Number of ensembled models", fontsize=13)
        plt.ylabel("Accuracy", fontsize=13)
        plt.tight_layout()

        plt.savefig('./results/figures/ensemble_' + data_name_arr[i] + '.pdf', format='pdf')
        plt.savefig('./results/figures/ensemble_' + data_name_arr[i] + '.png', format='png')
        plt.clf()


def plot_ensemble_multiclass_results():
    data_name_arr = ['BNCI2014001-4c']

    # BNCI2014001-4c
    arr1 = [[55.7, 56.31, 56.5, 56.66, 56.69, 56.7, 56.7, 56.48, 56.52], [54.44, 55.95, 56.24, 56.39, 56.44, 56.41, 56.47, 56.33, 56.23], [50.05, 56.52, 56.69, 56.99, 56.85, 56.97, 56.91, 56.83, 56.87]]
    arrs = [arr1]

    labels = ["averaging", "voting", "SML-soft"]

    for i in range(len(arrs)):
        arr = arrs[i]
        arr = np.array(arr)[:, 1:]

        y1 = arr[0]
        y2 = arr[1]
        y4 = arr[2]

        #x = np.arange(9).astype(int) + 2
        x = np.arange(8).astype(int) + 3

        plt.plot(x, y1, color='brown', marker='x', markersize=4, alpha=0.7, lw=2, label='Averaging')
        plt.plot(x, y2, color='green', marker='o', markersize=4, alpha=0.7, lw=2, label='Voting')
        plt.plot(x, y4, color='red', marker='^', markersize=4, alpha=0.7, lw=2, label='SML-soft')

        plt.grid(axis='y', color='grey', linestyle='--', lw=1, alpha=0.5)
        plt.legend(loc="lower right", fontsize=13)
        plt.xlabel("Number of ensembled models", fontsize=13)
        plt.ylabel("Accuracy", fontsize=13)
        plt.tight_layout()

        plt.savefig('./results/figures/ensemble_multiclass_' + 'ensemble_multiclass_' + data_name_arr[i] + '.pdf', format='pdf')
        plt.savefig('./results/figures/ensemble_multiclass_' + 'ensemble_multiclass_' + data_name_arr[i] + '.png', format='png')
        plt.clf()


def plot_line_graph_data_quantity():
    data_name_arr = ['BNCI2014001', 'BNCI2014002', 'BNCI2015001', 'BNCI2014008', 'BNCI2014009', 'BNCI2015003']

    arr1 = [69.676, 72.37644, 71.99067, 72.06789, 72.37656, 73.07089, 72.45356, 72.68511, 72.53067, 73.76533]
    arr2 = [69.0, 72.14286, 73.35714, 72.57143, 72.64286, 72.5, 72.07143, 72.21429, 72.78571, 72.71429]
    arr3 = [70.0, 71.16667, 70.66667, 71.875, 71.33333, 72.0, 73.04167, 72.29167, 71.5, 72.0]

    arr1net = [53.421, 64.377, 70.756, 72.582, 72.531, 72.145, 73.174, 73.688, 73.457, 72.942]
    arr2net = [56.381, 68.405, 71.429, 71.0, 72.619, 72.333, 73.119, 73.0, 72.571, 72.714]
    arr3net = [61.069, 67.861, 70.486, 71.0, 71.306, 71.417, 73.208, 72.611, 72.694, 73.042]

    arr4 = [65.20338, 67.509, 68.73212, 69.06425, 69.0805, 69.19987, 69.4305, 69.43225, 69.6965, 69.7305]
    arr5 = [71.4271, 73.9827, 75.5832, 76.118, 77.646, 77.5522, 78.3957, 78.9721, 78.8992, 79.5485]
    arr6 = [60.4095, 62.3233, 64.1305, 65.1702, 65.6515, 66.0492, 66.5937, 66.8333, 67.1145, 66.8934]

    arr4net = [68.809, 70.099, 70.488, 70.749, 71.132, 70.927, 71.166, 71.03, 70.827, 70.61]
    arr5net = [77.908, 79.244, 79.972, 80.291, 80.553, 80.69, 80.677, 80.926, 81.293, 81.243]
    arr6net = [63.846, 64.772, 65.792, 66.743, 66.427, 66.651, 67.609, 67.288, 67.818, 67.29]

    arrs = [arr1, arr2, arr3, arr4, arr5, arr6]
    arrsnet = [arr1net, arr2net, arr3net, arr4net, arr5net, arr6net]
    for i in range(len(arrs)):
    #for i in range(0, 3, 1):
        arr = arrs[i]
        arrnet = arrsnet[i]
        x = (np.arange(10).astype(int) + 1) * 10
        if data_name_arr[i] in ['BNCI2014001', 'BNCI2014002', 'BNCI2015001']:
            method = 'EA+CSP+LDA'
        elif data_name_arr[i] in ['BNCI2014008', 'BNCI2014009', 'BNCI2015003']:
            method = 'EA+xDAWN+LDA'

        #plt.plot(x, arr1, color='brown', marker='x', markersize=4, alpha=0.7, lw=2, label=data_name_arr[0])
        #plt.plot(x, arr2, color='green', marker='^', markersize=4, alpha=0.7, lw=2, label=data_name_arr[1])
        plt.plot(x, arr, color='green', marker='o', markersize=4, alpha=0.7, lw=2, label=method)
        plt.plot(x, arrnet, color='blue', marker='x', markersize=4, alpha=0.7, lw=2, label='EA+EEGNet')

        plt.title(data_name_arr[i])
        plt.grid(axis='y', color='grey', linestyle='--', lw=1, alpha=0.5)
        plt.legend(loc="lower right", fontsize=16)
        plt.xlabel("Percentage of Training Data Used", fontsize=16)
        if data_name_arr[i] in ['BNCI2014001', 'BNCI2014002', 'BNCI2015001']:
            label_str = 'Accuracy'
        elif data_name_arr[i] in ['BNCI2014008', 'BNCI2014009', 'BNCI2015003']:
            label_str = 'BCA'
        plt.ylabel("Test " + label_str, fontsize=16)
        plt.xticks([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        plt.tight_layout()

        #plt.savefig('./results/figures/ensemble_multiclass_' + 'ensemble_multiclass_' + data_name_arr[i] + '.pdf', format='pdf')
        plt.savefig('./results/figures/data_quantity_' + data_name_arr[i] + '.png', format='png')
        plt.clf()


def plot_line_graph_data_quantity_within():
    data_name_arr = ['BNCI2014001', 'BNCI2014002', 'BNCI2015001', 'BNCI2014008', 'BNCI2014009', 'BNCI2015003']

    arr1 = [57.86333, 68.19922, 68.84533, 68.56067, 73.14833, 73.94633, 75.25267, 77.03711]
    arr2 = [61.8255, 63.30357, 70.81629, 72.73814, 71.14286, 70.89286, 74.28579, 72.5]
    arr3 = [66.85183, 71.35417, 74.22617, 76.80558, 80.08333, 80.625, 82.5, 84.79167]
    arr4 = [65.02988, 69.10288, 71.78325, 73.47313, 75.40725, 76.17862, 77.23938, 77.23212]
    arr5 = [70.2323, 75.9173, 79.0553, 81.4287, 82.1874, 82.6156, 83.026, 83.2957]
    arr6 = [56.0608, 58.9918, 60.9301, 63.7908, 66.1811, 68.8458, 70.6578, 71.5158]

    arrs = [arr1, arr2, arr3, arr4, arr5, arr6]
    for i in range(len(arrs)):
    #for i in range(0, 3, 1):
        arr = arrs[i]
        x = (np.arange(8).astype(int) + 1) * 10

        if data_name_arr[i] in ['BNCI2014001', 'BNCI2014002', 'BNCI2015001']:
            method = 'CSP+LDA'
        elif data_name_arr[i] in ['BNCI2014008', 'BNCI2014009', 'BNCI2015003']:
            method = 'xDAWN+LDA'

        #plt.plot(x, arr1, color='brown', marker='x', markersize=4, alpha=0.7, lw=2, label=data_name_arr[0])
        #plt.plot(x, arr2, color='green', marker='^', markersize=4, alpha=0.7, lw=2, label=data_name_arr[1])
        plt.plot(x, arr, color='green', marker='o', markersize=4, alpha=0.7, lw=2, label=method)

        plt.title(data_name_arr[i])
        plt.grid(axis='y', color='grey', linestyle='--', lw=1, alpha=0.5)
        plt.legend(loc="lower right", fontsize=16)
        plt.xlabel("Percentage Used as Training Data", fontsize=16)
        if data_name_arr[i] in ['BNCI2014001', 'BNCI2014002', 'BNCI2015001']:
            label_str = 'Accuracy'
        elif data_name_arr[i] in ['BNCI2014008', 'BNCI2014009', 'BNCI2015003']:
            label_str = 'BCA'
        plt.ylabel("Test " + label_str, fontsize=16)
        plt.xticks([10, 20, 30, 40, 50, 60, 70, 80])
        plt.tight_layout()

        #plt.savefig('./results/figures/ensemble_multiclass_' + 'ensemble_multiclass_' + data_name_arr[i] + '.pdf', format='pdf')
        plt.savefig('./results/figures/data_quantity_within_' + data_name_arr[i] + '.png', format='png')
        plt.clf()


def plot_line_graph_GE():
    data_name_arr = ['BNCI2014001', 'BNCI2014002', 'BNCI2015001']#, 'BNCI2014008', 'BNCI2014009', 'BNCI2015003']

    arr1 = [72.99378, 73.22533, 73.30244, 72.99367, 72.22222]
    arr2 = [71.64286, 70.78571, 70.5, 72.21429, 70.57143]
    arr3 = [71.20833, 71.54167, 71.25, 70.75, 70.125]

    x = np.arange(5).astype(int) + 6

    plt.plot(x, arr1, color='blue', marker='o', markersize=4, alpha=0.7, lw=2, label=data_name_arr[0])
    plt.plot(x, arr2, color='green', marker='^', markersize=4, alpha=0.7, lw=2, label=data_name_arr[1])
    plt.plot(x, arr3, color='brown', marker='x', markersize=4, alpha=0.7, lw=2, label=data_name_arr[2])


    plt.title('GE with different k\', with k=10')
    plt.grid(axis='y', color='grey', linestyle='--', lw=1, alpha=0.5)
    plt.legend(loc="lower left", fontsize=12)
    plt.xlabel("k'", fontsize=16)
    plt.yticks([69, 70, 71, 72, 73, 74])
    plt.xticks([6, 7, 8, 9, 10])
    label_str = 'Accuracy'
    plt.ylabel("Test " + label_str, fontsize=16)
    plt.tight_layout()

    # plt.savefig('./results/figures/ensemble_multiclass_' + 'ensemble_multiclass_' + data_name_arr[i] + '.pdf', format='pdf')
    plt.savefig('./results/figures/data_quality_GE_MI.png', format='png')
    plt.clf()


def plot_tsne_subjects(dataset, align):
    X, y, num_subjects, paradigm, sample_rate, ch_num = data_loader(dataset)
    print('X, y, num_subjects, paradigm, sample_rate:', X.shape, y.shape, num_subjects, paradigm, sample_rate)

    if dataset == 'BNCI2014001':
        trial_num = 144
    elif dataset == 'BNCI2014002':
        trial_num = 100
    elif dataset == 'BNCI2015001':
        trial_num = 200
    elif dataset == 'BNCI2014008':
        trial_num = 4200
    elif dataset == 'BNCI2014009':
        trial_num = 1728
    elif dataset == 'BNCI2015003':
        trial_num = 2520

    if paradigm == 'ERP':
        print('ERP downsampled')
        X = mne.filter.resample(X, down=4)
        sample_rate = int(sample_rate // 4)
    print('sample rate:', sample_rate)

    if align:
        X = data_alignment(X, num_subjects)

    if paradigm == 'MI':
        # CSP
        csp = CSP(n_components=10)
        X = csp.fit_transform(X, y)
    elif paradigm == 'ERP':
        # xDAWN
        xdawn = Xdawn(n_components=X.shape[1])  # number of channels
        info = dataset_to_file(dataset, data_save=False)
        X = mne.EpochsArray(X, info)
        X = xdawn.fit_transform(X)
        X = X.reshape(X.shape[0], -1)

    X_2d = TSNE(n_components=2, learning_rate=10,
                init='random', random_state=42).fit_transform(X)  # 降维到2维

    labels_color = []

    # https://matplotlib.org/stable/gallery/color/named_colors.html
    color_arr = ['green', 'yellow',  'orange', 'violet',
                 'blue', 'dark blue', 'black', 'magenta', 'pink', 'red', 'orchid', 'azure', 'teal', 'aquamarine']

    for i in range(num_subjects):
        labels_color.append([color_arr[i]] * trial_num)
    labels_color = np.concatenate(labels_color)

    palette = []
    for i in range(num_subjects):
        palette.append(sns.xkcd_rgb[color_arr[i]])

    legend_arr = []
    for i in range(num_subjects):
        legend_arr.append('Subject' + str(i + 1))

    sns.scatterplot(x=X_2d[:, 0], y=X_2d[:, 1], s=3, hue=labels_color, legend=False, linewidth=0, palette=palette).set(title=dataset + ' EA=' + str(align))

    plt.savefig('./results/figures/' + dataset + '_EA-' + str(align) + '_TSNE.png')
    plt.clf()


def plot_tsne_classes(dataset, align):
    X, y, num_subjects, paradigm, sample_rate, ch_num = data_loader(dataset)
    print('X, y, num_subjects, paradigm, sample_rate:', X.shape, y.shape, num_subjects, paradigm, sample_rate)

    if dataset == 'BNCI2014001':
        trial_num = 144
    elif dataset == 'BNCI2014002':
        trial_num = 100
    elif dataset == 'BNCI2015001':
        trial_num = 200
    elif dataset == 'BNCI2014008':
        trial_num = 4200
    elif dataset == 'BNCI2014009':
        trial_num = 1728
    elif dataset == 'BNCI2015003':
        trial_num = 2520

    class_num = 2

    if paradigm == 'ERP':
        print('ERP downsampled')
        X = mne.filter.resample(X, down=4)
        sample_rate = int(sample_rate // 4)
    print('sample rate:', sample_rate)

    if align:
        X = data_alignment(X, num_subjects)

    if paradigm == 'MI':
        # CSP
        csp = CSP(n_components=10)
        X = csp.fit_transform(X, y)
    elif paradigm == 'ERP':
        # xDAWN
        xdawn = Xdawn(n_components=X.shape[1])  # number of channels
        info = dataset_to_file(dataset, data_save=False)
        X = mne.EpochsArray(X, info)
        X = xdawn.fit_transform(X)
        X = X.reshape(X.shape[0], -1)

    X_2d = TSNE(n_components=2, learning_rate=10,
                init='random', random_state=42).fit_transform(X)  # 降维到2维

    labels_color = []
    markers = []

    # https://matplotlib.org/stable/gallery/color/named_colors.html
    color_arr = ['green', 'yellow',  'orange', 'violet',
                 'blue', 'dark blue', 'black', 'magenta', 'pink', 'red', 'orchid', 'azure', 'teal', 'aquamarine']

    # TODO: ERP class imbalance split

    for i in range(num_subjects):
        for marker in ['x', 'o']:
            labels_color.append([color_arr[i]] * (trial_num // 2))
            markers.append(marker * (trial_num // 2))
    labels_color = np.concatenate(labels_color)

    palette = []
    for i in range(num_subjects):
        palette.append(sns.xkcd_rgb[color_arr[i]])

    legend_arr = []
    for i in range(num_subjects):
        legend_arr.append('Subject' + str(i + 1))

    sns.scatterplot(x=X_2d[:, 0], y=X_2d[:, 1], s=3, hue=labels_color, legend=False, markers=markers, linewidth=0, palette=palette).set(title=dataset + ' EA=' + str(align))

    plt.savefig('./results/figures/' + dataset + '_classes_EA-' + str(align) + '_TSNE.png')
    plt.clf()



if __name__ == '__main__':
    '''
    mat_data_path = '/Users/Riccardo/mne_data/MNE-bnci-data/database/data-sets/001-2014/A01T.mat'
    #mat_data_path = '/Users/Riccardo/Workspace/HUST-BCI/data/SEED/Raw/1_20131027.mat'

    config = {}
    config['path'] = mat_data_path
    config['x_name'] = 'data'
    config['sampling_rate'] = 256
    config['ch_names'] = ['Fz'] * 22
    config['description'] = 'MI dataset Subject 1'

    plot_eeg(config)
    '''

    #plot_mne_eeg()

    #plot_ensemble_results()
    #plot_ensemble_multiclass_results()

    #plot_line_graph_data_quantity()
    #plot_line_graph_data_quantity_within()
    #plot_line_graph_GE()

    for dataset in ['BNCI2014001', 'BNCI2014002', 'BNCI2015001']:
    #for dataset in ['BNCI2014008', 'BNCI2014009', 'BNCI2015003']:
        for align in [True, False]:
            plot_tsne_classes(dataset=dataset, align=align)
