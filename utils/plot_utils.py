import sys

import mne
import scipy
import numpy as np
import scipy.io
import matplotlib
import matplotlib.pyplot as plt

from mne.time_frequency import tfr_morlet, psd_multitaper, psd_welch
#from moabb.datasets import BNCI2014001

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
    print(data[:5, :5])

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


if __name__ == '__main__':


    #mat_data_path = '/Users/Riccardo/Workspace/HUST-BCI/data/BNCI-001-2014/A01T.mat'
    mat_data_path = '/Users/Riccardo/Workspace/HUST-BCI/data/SEED/Raw/1_20131027.mat'

    config = {}
    config['path'] = mat_data_path
    config['x_name'] = 'djc_eeg1'
    config['sampling_rate'] = 1000
    config['ch_names'] = ['FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'OZ', 'O2', 'CB2']
    config['description'] = 'SEED dataset Subject 1'

    plot_eeg(config)