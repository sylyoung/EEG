import sys

import mne
import scipy
import numpy as np
import scipy.io
import matplotlib
import matplotlib.pyplot as plt

from mne.time_frequency import tfr_morlet, psd_multitaper, psd_welch

from moabb.datasets import BNCI2014001, BNCI2014002, BNCI2014008, BNCI2014009, BNCI2015003, BNCI2015004, EPFLP300, BNCI2014004, BNCI2015001
from moabb.paradigms import MotorImagery, P300

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

    plot_mne_eeg()