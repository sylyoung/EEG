import numpy as np
import joblib

import mne


if __name__ == '__main__':

    save = False

    fs = 500  # 500 Hz
    time_win = 4  # 4 seconds window

    mne.set_log_level(verbose='ERROR')

    (X1, Y1) = joblib.load('/Users/Riccardo/Workspace/HUST-BCI/data/Music/data1106.jb')
    (X2, Y2) = joblib.load('/Users/Riccardo/Workspace/HUST-BCI/data/Music/data1108.jb')

    label_all = []
    label1 = Y1['label']
    label2 = Y2['label']
    label_all.append(label1)
    label_all.append(label2)
    label_all = np.concatenate(label_all)

    X_all = []
    for item in X1:
        X_all.append(np.array(item))
    for item in X2:
        X_all.append(np.array(item))

    trial_num = []

    cnt = 0
    subject_data = []
    subject_labels = []
    for item in X_all:
        item = np.transpose(item)
        ch_names = ['EEG1', 'EEG2', 'EEG3', 'EEG4', 'EEG5', 'EEG6', 'EEG7', 'EEG8']

        info = mne.create_info(ch_names=ch_names, sfreq=500, ch_types=['eeg'] * 8)

        # TODO: preprocessing

        # delete proj
        raw = mne.io.RawArray(item, info)
        raw.del_proj()

        # bandpass filter [1,50] Hz
        raw.filter(1, 50, n_jobs=8,  # use more jobs to speed up.
                   l_trans_bandwidth=1,  # make sure filter params are the same
                   h_trans_bandwidth=1,  # in each band and skip "auto" option.
                   verbose=None)
        data = raw.get_data()
        data = mne.filter.resample(data, down=2)  # downsample to 250Hz
        print(data.shape)

        # direct split
        # TODO: window function?
        a = data.shape[-1] % (250 * time_win)
        if data.shape[-1] % (250 * time_win) != 0.0:
            data = data[:, :-1 * a]
        data = np.split(data, indices_or_sections=data.shape[-1] // (250 * time_win), axis=-1)
        data = np.stack(data, axis=0)

        trial_num.append(len(data))

        label = label_all[cnt][0] / 90  # TODO:valence # divide by 90 because range is [0,90]?
        label = np.full((data.shape[0], 1), label)
        subject_data.append(data)
        subject_labels.append(label)

        cnt += 1

    subject_data = np.concatenate(subject_data, axis=0)
    subject_labels = np.concatenate(subject_labels, axis=0).reshape(-1, )

    print('subject/session data shape:', subject_data.shape, subject_labels.shape)

    if save:
        np.save('./data/' + 'Music' + '/X', subject_data)
        np.save('./data/' + 'Music' + '/y', subject_labels)

    print(trial_num)  # TODO: This array is directly copied to be used in main. Write a function instead.
