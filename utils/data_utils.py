import os
import sys

import numpy as np
import scipy.io as sio
import moabb
import mne
import pickle

from moabb.datasets import BNCI2014001, BNCI2014002, BNCI2014008, BNCI2014009, BNCI2015003, BNCI2015004, EPFLP300, BNCI2014004, BNCI2015001, PhysionetMI, Cho2017, Wang2016
from moabb.paradigms import MotorImagery, P300, SSVEP
from scipy.stats import differential_entropy
from scipy.signal import stft
from pykalman import KalmanFilter

lib_path = os.path.abspath(os.path.join('..'))
sys.path.append(lib_path)

from EEG.utils.alg_utils import EA
from EEG.utils.feature_utils import de_feature_extractor


def split_data(data, axis, times):
    # Splitting data into multiple sections. data: (trials, channels, time_samples)
    data_split = np.split(data, indices_or_sections=times, axis=axis)
    return data_split


def convert_label(labels, axis, threshold):
    # Converting labels to 0 or 1, based on a certain threshold
    label_01 = np.where(labels > threshold, 1, 0)
    #print(label_01)
    return label_01


def time_cut(data, cut_percentage):
    # Time Cutting: cut at a certain percentage of the time. data: (..., ..., time_samples)
    data = data[:, :, :int(data.shape[2] * cut_percentage)]
    return data


def data_loader_DEAP(data_folder):

    def sort_func(name_string):
        if 'DS_Store' in name_string:
            return -1
        if name_string.endswith('.mat'):
            id_ = name_string[:-4]
        if id_.startswith('s'):
            id_ = id_[1:]
        if id_.startswith('0'):
            id_ = id_[1:]
        return int(id_)

    for subdir, dirs, files in os.walk(data_folder):
        for file in sorted(files, key=sort_func):
            f_path = os.path.join(subdir, file)
            if 'DS_Store' in f_path:
                continue

    file_arr = os.listdir(data_folder)
    for file_name in file_arr:
        if not file_name.endswith('.mat'):
            print('removing', file_name)
            file_arr.remove(file_name)
    subj_num = len(file_arr)
    print('DATA_PATH:', data_folder)
    print('SUBJECT_NUM:', subj_num)

    data = []
    labels = []
    for i in range(1, subj_num + 1):
        try:
            mat = sio.loadmat(data_folder + "/" + 's' + str(i) + ".mat")
        except:
            mat = sio.loadmat(data_folder + "/" + 's0' + str(i) + ".mat")
        x = np.array(mat['data'])
        # only EEG channels
        x = x[:, :32, :]
        y = np.moveaxis(np.array(mat['labels']), -1, 0)
        y = convert_label(y, 0, 5.0)
        y = y.reshape(-1, 1)
        print(x.shape, y.shape)

        data.append(x)
        labels.append(y)

    return data, labels


def traintest_split_cross_subject(dataset, X, y, num_subjects, test_subject_id, trial_num_arr=None):
    if trial_num_arr is None:
        X = np.split(X, indices_or_sections=num_subjects, axis=0)
        y = np.split(y, indices_or_sections=num_subjects, axis=0)
        test_x = X.pop(test_subject_id)
        test_y = y.pop(test_subject_id)
        train_x = np.concatenate(X, axis=0)
        train_y = np.concatenate(y, axis=0)
    else:
        X_c = X.copy()
        y_c = y.copy()
        test_x = X_c.pop(test_subject_id)
        test_y = y_c.pop(test_subject_id)
        train_x = np.concatenate(X_c, axis=0)
        train_y = np.concatenate(y_c, axis=0)

    print('Test subject s' + str(test_subject_id))
    print('Training/Test split:', train_x.shape, test_x.shape)
    return train_x, train_y, test_x, test_y


def traintest_split_cross_subject_uneven_multiple(dataset, X, y, num_subjects, test_subject_id, trial_num_arr):
    data_subjects = []
    labels_subjects = []
    for i in range(num_subjects):
        for j in range(len(trial_num_arr[0])):
            data_subjects.append(X[int(np.sum(trial_num_arr[:i])) + int(np.sum(trial_num_arr[i, :j])):np.sum(trial_num_arr[:i]) + int(np.sum(trial_num_arr[i, :j+1]))])
            labels_subjects.append(y[int(np.sum(trial_num_arr[:i])) + int(np.sum(trial_num_arr[i, :j])):np.sum(trial_num_arr[:i]) + int(np.sum(trial_num_arr[i, :j+1]))])

    test_x = []
    test_y = []
    for j in range(len(trial_num_arr[0])):
        test_x.append(data_subjects.pop(test_subject_id * len(trial_num_arr[0])))
        test_y.append(labels_subjects.pop(test_subject_id * len(trial_num_arr[0])))
    test_x = np.concatenate(test_x, axis=0)
    test_y = np.concatenate(test_y, axis=0)
    train_x = np.concatenate(data_subjects, axis=0)
    train_y = np.concatenate(labels_subjects, axis=0)
    print('Test subject s' + str(test_subject_id))
    print('Training/Test split:', train_x.shape, test_x.shape)
    return train_x, train_y, test_x, test_y


def dataset_SEED_extracted_process(data_folder):

    def sort_func(name_string):
        id_ = -1
        if name_string.endswith('.mat') and not name_string.startswith('label'):
            if name_string[1] == '_':
                id_ = int(name_string[:1]) * (2 ** 28) + int(name_string[-12:-4])
            else:
                id_ = int(name_string[:2]) * (2 ** 28) + int(name_string[-12:-4])
        return id_

    file_arr = []
    for subdir, dirs, files in os.walk(data_folder):
        for file in sorted(files, key=sort_func):
            f_path = os.path.join(subdir, file)
            if 'DS_Store' in f_path or 'label' in f_path or not '.mat' in f_path:
                continue
            print(f_path)
            file_arr.append(f_path)

    subj_num = int(len(file_arr) / 3)
    print('DATA_PATH:', data_folder)
    print('SUBJECT_NUM:', subj_num)

    label_arr = np.array([1,0,-1,-1,0,1,-1,0,1,1,0,-1,0,1,-1]) + 1

    data = []
    labels = []
    for i in range(len(file_arr)):
        print(file_arr[i])
        mat = sio.loadmat(file_arr[i])
        subject_data = []
        subject_label = []
        for j in range(15):
            x = np.array(mat['de_LDS' + str(j + 1)])
            y = np.full((x.shape[1], 1), label_arr[j]).reshape(-1,)
            subject_data.append(x)
            subject_label.append(y)
        subject_data = np.concatenate(subject_data, axis=1)
        subject_label = np.concatenate(subject_label, axis=0)
        print(subject_data.shape, subject_label.shape)
        data.append(subject_data)
        labels.append(subject_label)
    data = np.concatenate(data, axis=1)
    labels = np.concatenate(labels, axis=0)
    print(data.shape, labels.shape)
    data = np.transpose(data, (1,0,2))
    print(data.shape, labels.shape)
    data = data.reshape(data.shape[0], -1)
    print(data.shape, labels.shape)
    np.save('./data/' + 'SEED' + '/X_extracted', data)
    np.save('./data/' + 'SEED' + '/labels_extracted', labels)


def dataset_SEEDV_extracted_process(data_folder):

    def sort_func(name_string):
        id_ = -1
        if name_string.endswith('.npz'):
            id_ = int(name_string[0])
            if name_string[1] != '_':
                id_ = int(name_string[:2])
        return id_

    file_arr = []
    for subdir, dirs, files in os.walk(data_folder):
        for file in sorted(files, key=sort_func):
            f_path = os.path.join(subdir, file)
            if 'DS_Store' in f_path or 'ipynb' in f_path or 'README' in f_path:
                continue
            print(f_path)
            file_arr.append(f_path)

    subj_num = int(len(file_arr))
    print('DATA_PATH:', data_folder)
    print('SUBJECT_NUM:', subj_num)

    data = []
    labels = []
    for i in range(len(file_arr)):
        print(file_arr[i])

        data_npz = np.load(file_arr[i])
        data_pkl = pickle.loads(data_npz['data'])
        label_pkl = pickle.loads(data_npz['label'])

        label_dict = {0: 'Disgust', 1: 'Fear', 2: 'Sad', 3: 'Neutral', 4: 'Happy'}
        for i in range(45):
            print(
                'Session {} -- Trial {} -- EmotionLabel : {}'.format(i // 15 + 1, i % 15 + 1, label_dict[label_pkl[i][0]]))

            #data_tmp = np.transpose(data_pkl[i], (1, 0))
            data_tmp = data_pkl[i]
            label_tmp = np.full((data_tmp.shape[0], 1), label_pkl[i][0]).reshape(-1, )
            print(data_tmp.shape, label_tmp.shape)

            data.append(data_tmp)
            labels.append(label_tmp)
    data = np.concatenate(data, axis=0)
    labels = np.concatenate(labels, axis=0)
    labels = labels.reshape(-1, )
    print(data.shape, labels.shape)
    input('')

    np.save('./data/' + 'SEED-V' + '/X_extracted', data)
    np.save('./data/' + 'SEED-V' + '/labels_extracted', labels)


def dataset_SEEDV_eye_extracted_process(data_folder):

    def sort_func(name_string):
        id_ = -1
        if name_string.endswith('.npz'):
            id_ = int(name_string[0])
            if name_string[1] != '_':
                id_ = int(name_string[:2])
        return id_

    file_arr = []
    for subdir, dirs, files in os.walk(data_folder):
        for file in sorted(files, key=sort_func):
            f_path = os.path.join(subdir, file)
            if 'DS_Store' in f_path or 'ipynb' in f_path or 'README' in f_path:
                continue
            print(f_path)
            file_arr.append(f_path)

    subj_num = int(len(file_arr))
    print('DATA_PATH:', data_folder)
    print('SUBJECT_NUM:', subj_num)

    data = []
    labels = []
    for i in range(len(file_arr)):
        print(file_arr[i])

        data_npz = np.load(file_arr[i])
        data_pkl = pickle.loads(data_npz['data'])
        label_pkl = pickle.loads(data_npz['label'])

        label_dict = {0: 'Disgust', 1: 'Fear', 2: 'Sad', 3: 'Neutral', 4: 'Happy'}
        for i in range(45):
            print(
                'Session {} -- Trial {} -- EmotionLabel : {}'.format(i // 15 + 1, i % 15 + 1, label_dict[label_pkl[i][0]]))

            #data_tmp = np.transpose(data_pkl[i], (1, 0))
            data_tmp = data_pkl[i]
            label_tmp = np.full((data_tmp.shape[0], 1), label_pkl[i][0]).reshape(-1, )
            print(data_tmp.shape, label_tmp.shape)

            data.append(data_tmp)
            labels.append(label_tmp)
    data = np.concatenate(data, axis=0)
    labels = np.concatenate(labels, axis=0)
    labels = labels.reshape(-1, )
    print(data.shape, labels.shape)
    input('')

    np.save('./data/' + 'SEED-V' + '/X_eye_extracted', data)
    np.save('./data/' + 'SEED-V' + '/labels_eye_extracted', labels)


def data_loader_split(dataset, test_subj_id, shuffle):

    data_folder = './data/' + str(dataset)

    '''
    def sort_func(name_string):
        if 'DS_Store' in name_string:
            return -1
        if name_string.endswith('.mat'):
            id_ = name_string[:-4]
        if name_string.startswith('s'):
            id_ = name_string[1:]
        return int(id_)

    for subdir, dirs, files in os.walk(data_folder):
        for file in sorted(files, key=sort_func):
            f_path = os.path.join(subdir, file)
            if 'DS_Store' in f_path:
                continue
    '''
    file_arr = os.listdir(data_folder)
    for file_name in file_arr:
        if not file_name.endswith('.mat'):
            print('removing', file_name)
            file_arr.remove(file_name)
    subj_num = len(file_arr)
    print('DATA_PATH:', data_folder)
    print('SUBJECT_NUM:', subj_num)
    print('TEST_SUBJ_ID:', test_subj_id)

    mat = sio.loadmat(data_folder + "/" + 's' + str(test_subj_id) + ".mat")
    x = np.moveaxis(np.array(mat['X']), -1, 0)
    y = np.array(mat['y'])
    test_x = x
    test_y = y

    train_x_arr = []
    train_y_arr = []

    for i in range(subj_num):
        if i == test_subj_id:
            continue
        train_x = []
        train_y = []
        mat = sio.loadmat(data_folder + "/" + 's' + str(i) + ".mat")
        x = np.moveaxis(np.array(mat['X']), -1, 0)
        y = np.array(mat['y'])

        train_x.append(x)
        train_y.append(y)
        train_x_arr.append(train_x)
        train_y_arr.append(train_y)

    train_x_array_out = []
    train_y_array_out = []
    valid_x_array_out = []
    valid_y_array_out = []

    print('VALID SET SPLITTING (subject-wise) (8:2)')
    for train_x, train_y in zip(train_x_arr, train_y_arr):
        train_x = np.concatenate(train_x, axis=0)
        train_y = np.concatenate(train_y, axis=0)

        np.random.seed(42)
        idx = list(range(len(train_y)))
        if shuffle:
            np.random.shuffle(idx)
        train_x = train_x[idx]
        train_y = train_y[idx]

        valid_x = train_x[int(len(train_x) * 0.8):]
        valid_y = train_y[int(len(train_y) * 0.8):]

        train_x = train_x[:int(len(train_x) * 0.8)]
        train_y = train_y[:int(len(train_y) * 0.8)]

        train_x_array_out.append(train_x)
        train_y_array_out.append(train_y)

        valid_x_array_out.append(valid_x)
        valid_y_array_out.append(valid_y)


    train_x = train_x_array_out
    train_y = train_y_array_out
    valid_x = valid_x_array_out
    valid_y = valid_y_array_out
    test_x = test_x
    test_y = test_y

    data = {}
    data['train_x'] = train_x
    data['train_y'] = train_y
    data['valid_x'] = valid_x
    data['valid_y'] = valid_y
    data['test_x'] = test_x
    data['test_y'] = test_y

    print('TRAIN_X number and shape:', len(train_x), ',', train_x[0].shape)
    print('TRAIN_y number and shape:', len(train_y), ',', train_y[0].shape)
    print('VALID_X number and shape:', len(valid_x), ',', valid_x[0].shape)
    print('VALID_y number and shape:', len(valid_y), ',', valid_y[0].shape)
    print('TEST_X shape:', test_x.shape)
    print('TEST_y shape:', test_y.shape)

    return data


def dataset_ERN_to_file(data_folder):

    dataset_name = 'ERN'

    def sort_func(name_string):
        if 'DS_Store' in name_string:
            return -1
        if name_string.endswith('.mat'):
            id_ = name_string[:-4]
        if id_.startswith('s'):
            id_ = id_[1:]
        return int(id_)

    file_arr = []
    for subdir, dirs, files in os.walk(data_folder):
        for file in sorted(files, key=sort_func):
            f_path = os.path.join(subdir, file)
            if 'DS_Store' in f_path:
                continue
            file_arr.append(f_path)

    subj_num = len(file_arr)
    print('DATA_PATH:', data_folder)
    print('SUBJECT_NUM:', subj_num)

    data = []
    labels = []
    for i in range(subj_num):
        print(file_arr[i])
        mat = sio.loadmat(file_arr[i])
        X = np.array(mat['x'])
        X = np.transpose(X, (2, 0, 1))
        y = np.array(mat['y']).reshape(-1, )
        print(X.shape, y.shape)
        data.append(X)
        labels.append(y)

    data = np.concatenate(data, axis=0)
    labels = np.concatenate(labels, axis=0)
    print(data.shape, labels.shape)
    np.save('./data/' + dataset_name + '/X', data)
    np.save('./data/' + dataset_name + '/labels', labels)


def dataset_SEED_DE_to_file(data_folder, use_EA):
    # 200 Hz
    # Bandpass 0-75 Hz
    # (152730, 62, 5) (152730,) 15subjects 3sessions 3394seconds, 62 channels, 5 DE features

    mne.set_log_level(verbose='ERROR')

    def sort_func(name_string):
        id_ = -1
        if name_string.endswith('.mat') and not name_string.startswith('label'):
            if name_string[1] == '_':
                id_ = int(name_string[:1]) * (2 ** 28) + int(name_string[-12:-4])
            else:
                id_ = int(name_string[:2]) * (2 ** 28) + int(name_string[-12:-4])
        return id_

    # channel names in order
    ch_names = ['FP1','FPZ','FP2','AF3','AF4','F7','F5','F3','F1','FZ','F2','F4','F6','F8','FT7','FC5','FC3','FC1',
                'FCZ','FC2','FC4','FC6','FT8','T7','C5','C3','C1','CZ','C2','C4','C6','T8','TP7','CP5','CP3','CP1',
                'CPZ','CP2','CP4','CP6','TP8','P7','P5','P3','P1','PZ','P2','P4','P6','P8','PO7','PO5','PO3','POZ',
                'PO4','PO6','PO8','CB1','O1','OZ','O2','CB2']

    # five eeg frequency bands
    iter_freqs = [
        ('Delta', 1, 4),
        ('Theta', 4, 8),
        ('Alpha', 8, 14),
        ('Beta', 14, 31),
        ('Gamma', 31, 50)
    ]
    '''
    # five eeg frequency bands
    iter_freqs_mod = [
        ('Delta', 0.5, 3.99),
        ('Theta', 4, 7.99),
        ('Alpha', 8, 13.99),
        ('Beta', 14, 34.99),
        ('Gamma', 35, 50)
    ]
    '''
    label_arr = np.array([1,0,-1,-1,0,1,-1,0,1,1,0,-1,0,1,-1]) + 1
    X = []
    labels = []
    subj_trials_num = []
    for subdir, dirs, files in os.walk(data_folder):
        for file in sorted(files, key=sort_func):
            f_path = os.path.join(subdir, file)
            if not '.mat' in f_path or 'label' in f_path:
                continue
            print('#' * 30)
            print(f_path)
            mat = sio.loadmat(f_path)
            subject_data = []
            subject_labels = []
            ind = 0
            for key, value in mat.items():
                if 'eeg' in key:
                    print(key)
                    data = value
                    label = label_arr[ind]
                    print('data shape and label:', data.shape, label)
                    '''
                    #for i in range(62):
                        #data_ = data[i].reshape(-1)
                    print(data.shape)
                    f, t, Zxx = stft(data, fs=200, window='hann', nperseg=200, noverlap=0, boundary=None,
                                     padded=False, axis=-1)
                    print(f.shape)
                    print(f)
                    print(t.shape)
                    print(Zxx.shape)
                    data = Zxx[0, 1:4, 0]
                    #print(Zxx)
                    print(data.shape)
                    DE = differential_entropy(data)
                    print(DE)
                    #print(data_)
                    sys.exit(0)
                    
                    data = de_feature_extractor(data)
                    print('Hanning data DE:', data.shape)
                    print('#' * 20 + '62x5 channels' + '#' * 20)
                    print(data[0, :])
                    print('#' * 20 + '235 timesamples' + '#' * 20)
                    print(data[:, 0])
                    sys.exit(0)
                    '''
                    # bandpass filter for every frequency band
                    info = mne.create_info(ch_names=ch_names, sfreq=200, ch_types=['eeg'] * 62)

                    raw = mne.io.RawArray(data, info)
                    data_freqs = []
                    for i in range(len(iter_freqs)):
                        print('Filter frequency band from ', iter_freqs[i][1], 'Hz to ', iter_freqs[i][2], 'Hz')
                        raw.filter(iter_freqs[i][1], iter_freqs[i][2], n_jobs=8,  # use more jobs to speed up.
                                   l_trans_bandwidth=1,  # make sure filter params are the same
                                   h_trans_bandwidth=1,  # in each band and skip "auto" option.
                                   verbose=None)
                        data_freq = raw.get_data()

                        data_freqs.append(data_freq)
                    data = np.stack(data_freqs, axis=0)

                    data = data[:, :, :-1]
                    a = data.shape[-1] % 800
                    if data.shape[-1] % 800 != 0.0:
                        data = data[:, :, :-1 * a]
                    data = split_data(data, axis=-1, times=data.shape[-1] // 800)
                    data = np.stack(data, axis=0)
                    label = np.full((data.shape[0], 1), label)
                    subject_data.append(data)
                    subject_labels.append(label)
                    print('trial splitted shape:', data.shape, label.shape)
                    ind += 1

            subject_data = np.concatenate(subject_data, axis=0)
            subject_labels = np.concatenate(subject_labels, axis=0).reshape(-1, )
            print('subject/session data shape:', subject_data.shape, subject_labels.shape)  # (3394, 5, 62, 200) (3394,)

            # EA
            if use_EA:
                for i in range(subject_data.shape[1]):
                    subject_data[:, i, :, :] = EA(subject_data[:, i, :, :])
                    print('EA\'d subject data shape:', subject_data[:, i, :, :].shape)

            # DE feature extraction
            features = []
            for i in range(subject_data.shape[1]):

                feature = differential_entropy(subject_data[:, i, :, :], axis=-1, method='vasicek')

                features.append(feature)
            features = np.concatenate(features, axis=-1)
            print('DE feature shape:', features.shape, subject_labels.shape)  # (3394, 310) (3394,)

            X.append(features)
            labels.append(subject_labels)
            subj_trials_num.append(len(subject_labels))

    X = np.concatenate(X, axis=0)
    labels = np.concatenate(labels, axis=0).reshape(-1,)
    labels = labels.reshape(-1, )
    subj_trials_num = np.array(subj_trials_num)
    print(X.shape)  # (152730, 310)
    print(labels.shape)  # (152730,)
    print(subj_trials_num)  # 3394
    if use_EA:
        np.save('./data/' + 'SEED' + '/new_X_DE_EA_4sec', X)
        np.save('./data/' + 'SEED' + '/new_labels_EA_4sec', labels)
    else:
        np.save('./data/' + 'SEED' + '/new_X_DE_4sec', X)
        np.save('./data/' + 'SEED' + '/new_labels_4sec', labels)


def dataset_SEEDV_DE_to_file(data_folder, use_EA):
    # 200 Hz
    # Bandpass 1-75 Hz
    # (152730, 62, 5) (152730,) 15subjects 3sessions 3394seconds, 62 channels, 5 DE features

    mne.set_log_level(verbose='ERROR')


    def sort_func(name_string):
        id_ = -1
        if name_string.endswith('.cnt'):
            if name_string[1] == '_':
                id_ = int(name_string[0]) * (2 ** 26) + int(name_string[2]) * (2 ** 20)
            elif name_string[2] == '_':
                id_ = int(name_string[:2]) * (2 ** 26) + int(name_string[3]) * (2 ** 20)
        return id_

    # channel names in order
    ch_names = ['FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5',
                'FC3', 'FC1',
                'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5',
                'CP3', 'CP1',
                'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5',
                'PO3', 'POZ',
                'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'OZ', 'O2', 'CB2']

    # five eeg frequency bands
    iter_freqs = [
        ('Delta', 1, 4),
        ('Theta', 4, 8),
        ('Alpha', 8, 14),
        ('Beta', 14, 31),
        ('Gamma', 31, 50)
    ]
    fStart = [1, 4, 8, 14, 31]
    fEnd = [4, 8, 14, 31, 50]
    '''
    # five eeg frequency bands
    iter_freqs_mod = [
        ('Delta', 0.5, 3.99),
        ('Theta', 4, 7.99),
        ('Alpha', 8, 13.99),
        ('Beta', 14, 34.99),
        ('Gamma', 35, 50)
    ]
    '''
    label_arr = np.array([4, 1, 3, 2, 0, 4, 1, 3, 2, 0, 4, 1, 3, 2, 0])  # session 1
    X = []
    labels = []
    subj_trials_num = []

    start_second1 = [30, 132, 287, 555, 773, 982, 1271, 1628, 1730, 2025, 2227, 2435, 2667, 2932, 3204]
    end_second1 = [102, 228, 524, 742, 920, 1240, 1568, 1697, 1994, 2166, 2401, 2607, 2901, 3172, 3359]

    start_second2 = [30, 299, 548, 646, 836, 1000, 1091, 1392, 1657, 1809, 1966, 2186, 2333, 2490, 2741]
    end_second2 = [267, 488, 614, 773, 967, 1059, 1331, 1622, 1777, 1908, 2153, 2302, 2428, 2709, 2817]

    start_second3 = [30, 353, 478, 674, 825, 908, 1200, 1346, 1451, 1711, 2055, 2307, 2457, 2726, 2888]
    end_second3 = [321, 418, 643, 764, 877, 1147, 1284, 1418, 1679, 1996, 2275, 2425, 2664, 2857, 3066]

    for subdir, dirs, files in os.walk(data_folder):
        for file in sorted(files, key=sort_func):
            f_path = os.path.join(subdir, file)
            if not '.cnt' in f_path:
                continue
            print('#' * 30)
            print(f_path)

            raw = mne.io.read_raw_cnt(f_path, preload=True)

            useless_ch = ['M1', 'M2', 'VEO', 'HEO']
            raw.drop_channels(useless_ch)

            data_mat = raw.get_data()
            print('loaded data shape:', data_mat.shape)

            subject_data = []
            subject_labels = []

            sample_freq = 1000

            if '_1_' in f_path:
                start_second = start_second1
                end_second = end_second1
                print('session 1')
            elif '_2_' in f_path:
                start_second = start_second2
                end_second = end_second2
                print('session 2')
            elif '_3_' in f_path:
                start_second = start_second3
                end_second = end_second3
                print('session 3')

            for j in range(15):

                data = data_mat[:, start_second[j] * 1000: end_second[j] * 1000]
                #print(data.shape)

                a = data.shape[-1] % 4000  # 4/1 seconds chunks
                if data.shape[-1] % 4000 != 0.0:
                    data = data[:, :-1 * a]
                data = split_data(data, axis=-1, times=data.shape[-1] // 4000)
                data = np.stack(data, axis=0)
                print(data.shape)

                label = label_arr[j]
                #print('label:', label)
                label = np.full((data.shape[0], 1), label)
                #print('label:', label.shape)

                '''
                info = mne.create_info(ch_names=ch_names, sfreq=sample_freq, ch_types=['eeg'] * 62)
                raw = mne.io.RawArray(data_trial, info)

                # bandpass filter for every frequency band
                data_freqs = []
                for i in range(len(iter_freqs)):
                    #print('Filter frequency band from ', iter_freqs[i][1], 'Hz to ', iter_freqs[i][2], 'Hz')
                    raw.filter(iter_freqs[i][1], iter_freqs[i][2], n_jobs=8,  # use more jobs to speed up.
                               l_trans_bandwidth=1,  # make sure filter params are the same
                               h_trans_bandwidth=1,  # in each band and skip "auto" option.
                               verbose=None)
                    data_freq = raw.get_data()
                    data_freq_downsample = mne.filter.resample(data_freq, down=5)
                    data_freqs.append(data_freq_downsample)
                data = np.stack(data_freqs, axis=0)
                
                data = data[:, :, :-1]
                a = data.shape[-1] % 200  # 4/1 seconds chunks
                if data.shape[-1] % 200 != 0.0:
                    data = data[:, :, :-1 * a]
                data = split_data(data, axis=-1, times=data.shape[-1] // 200)
                data = np.stack(data, axis=0)
                label = label_arr[j]
                print('label:', label)
                label = np.full((data.shape[0], 1), label)
                '''

                subject_data.append(data)
                subject_labels.append(label)
            subject_data = np.concatenate(subject_data, axis=0)
            subject_labels = np.concatenate(subject_labels, axis=0).reshape(-1, )
            print('subject/session data shape:', subject_data.shape, subject_labels.shape)  # (681, 62, 4000) (681,)

            # EA
            if use_EA:
                subject_data = EA(subject_data)
                print('EA\'d')

            '''
            # DE feature extraction by Xueliang
            feature = []
            for i in range(subject_data.shape[0]):
                data = de_feature_extractor(subject_data[i], fs=sample_freq, fStart=fStart, fEnd=fEnd, window=4, stftn=512)
                feature.append(data)
            feature = np.concatenate(feature, axis=0)
            print('feature.shape, labels.shape:', feature.shape, subject_labels.shape)
            '''

            '''
            # DE feature extraction
            features = []
            for i in range(subject_data.shape[1]):
                feature = differential_entropy(subject_data[:, i, :, :], axis=-1, method='vasicek')
                # feature = test_method(subject_data[:, i, :, :], axis=-1)
                features.append(feature)
            features = np.concatenate(features, axis=-1)
            print('DE feature shape:', features.shape, subject_labels.shape)  # (3394, 310) (3394,)
            '''
            print(subject_data.shape, subject_labels.shape)
            X.append(subject_data)
            #X.append(feature)
            labels.append(subject_labels)
            subj_trials_num.append(len(subject_labels))

    X = np.concatenate(X, axis=0)
    labels = np.concatenate(labels, axis=0).reshape(-1, )
    labels = labels.reshape(-1, )
    subj_trials_num = np.array(subj_trials_num)
    print(X.shape)  # (152730, 310)
    print(labels.shape)  # (152730,)
    print(subj_trials_num)  # 3394
    if use_EA:
        np.save('./data/' + 'SEED-V' + '/X_EA_4sec', X)
        np.save('./data/' + 'SEED-V' + '/labels_EA_4sec', labels)
    else:
        np.save('./data/' + 'SEED-V' + '/X_4sec', X)
        np.save('./data/' + 'SEED-V' + '/labels_4sec', labels)


def dataset_DEAP_DE_to_file(data_folder, use_EA):
    # 128 Hz
    # Bandpass 4-45 Hz
    # (8064 * 40 * 32, 40, 5) (8064 * 40 * 32, 4) 32subjects 40trials 63seconds, 40 channels, 5 DE features; 4 dimensions

    mne.set_log_level(verbose='ERROR')

    def sort_func(name_string):
        id_ = -1
        if name_string.endswith('.mat'):
            id_ = int(name_string[1])
            if name_string[2] != '_':
                id_ = int(name_string[1:3])
        return id_

    # channel names in order
    ch_names = ['Fp1','AF3','F3','F7','FC5','FC1','C3','T7','CP5','CP1','P3','P7','PO3','O1','Oz','Pz','Fp2','AF4',
                'Fz','F4','F8','FC6','FC2','Cz','C4','T8','CP6','CP2','P4','P8','PO4','O2'] #,'hEOG','vEOG','zEMG','tEMG',
                #'GSR','Respiration_belt','Plethysmograph','Temperature']
    ch_types = ['eeg'] * 32
    #ch_types.extend(['eog', 'eog', 'emg', 'emg', 'misc', 'misc', 'misc', 'misc'])

    # five eeg frequency bands
    iter_freqs = [
        ('Delta', 1, 3),
        ('Theta', 4, 7),
        ('Alpha', 8, 13),
        ('Beta', 14, 30),
        ('Gamma', 31, 50)
    ]
    '''
    # five eeg frequency bands
    iter_freqs_mod = [
        ('Delta', 0.5, 3.99),
        ('Theta', 4, 7.99),
        ('Alpha', 8, 13.99),
        ('Beta', 14, 34.99),
        ('Gamma', 35, 50)
    ]
    '''

    X = []
    labels = []
    subj_trials_num = []
    for subdir, dirs, files in os.walk(data_folder):
        for file in sorted(files, key=sort_func):
            f_path = os.path.join(subdir, file)
            if not '.mat' in f_path:
                continue
            print('#' * 30)
            print(f_path)

            mat = sio.loadmat(f_path)
            mat_data = mat['data'] # (40, 40, 8064)
            mat_label = mat['labels']

            subject_data = []
            subject_labels = []

            for i in range(mat_data.shape[0]):

                # bandpass filter for every frequency band
                info = mne.create_info(ch_names=ch_names, sfreq=128, ch_types=ch_types)
                raw = mne.io.RawArray(mat_data[i, :32, :(-3 * 128)], info)
                data_freqs = []
                for i in range(len(iter_freqs)):
                    #print('Filter frequency band from ', iter_freqs[i][1], 'Hz to ', iter_freqs[i][2], 'Hz')
                    raw.filter(iter_freqs[i][1], iter_freqs[i][2], n_jobs=8,  # use more jobs to speed up.
                               l_trans_bandwidth=1,  # make sure filter params are the same
                               h_trans_bandwidth=1,  # in each band and skip "auto" option.
                               verbose=None)
                    data_freq = raw.get_data()
                    data_freqs.append(data_freq)
                data = np.stack(data_freqs, axis=0)

                #data = data[:, :, :]
                a = data.shape[-1] % 200
                if data.shape[-1] % 200 != 0.0:
                    data = data[:, :, :-1 * a]
                data = split_data(data, axis=-1, times=data.shape[-1] // 200)
                data = np.stack(data, axis=0)
                label = np.tile(mat_label[i], (data.shape[0], 1))
                #print('trial splitted shape:', data.shape, label.shape)
                subject_data.append(data)
                subject_labels.append(label)

            subject_data = np.concatenate(subject_data, axis=0)
            subject_labels = np.concatenate(subject_labels, axis=0)
            print('subject/session data shape:', subject_data.shape, subject_labels.shape)  # (1520, 5, 32, 200) (1520, 4)

            # EA
            if use_EA:
                for i in range(subject_data.shape[1]):
                    subject_data[:, i, :, :] = EA(subject_data[:, i, :, :])
                    print('EA\'d subject data shape:', subject_data[:, i, :, :].shape)

            # DE feature extraction
            features = []
            for i in range(subject_data.shape[1]):
                feature = differential_entropy(subject_data[:, i, :, :], axis=-1, method='vasicek')
                features.append(feature)
            features = np.concatenate(features, axis=-1)
            print('DE feature shape:', features.shape, subject_labels.shape)  # (3394, 310) (3394,)

            X.append(features)
            labels.append(subject_labels)
            subj_trials_num.append(len(subject_labels))

    X = np.concatenate(X, axis=0)
    labels = np.concatenate(labels, axis=0).reshape(-1, 4)
    subj_trials_num = np.array(subj_trials_num)
    print(X.shape)  # (48640, 160)
    print(labels.shape)  # (48640, 4)
    print(subj_trials_num)  # 1520

    if use_EA:
        np.save('./data/' + 'DEAP' + '/X_DE_EA', X)
        np.save('./data/' + 'DEAP' + '/labels_EA', labels)
    else:
        np.save('./data/' + 'DEAP' + '/X_DE', X)
        np.save('./data/' + 'DEAP' + '/labels', labels)


def feature_smooth_moving_average(X, ma_window, subject_num, session_num, trial_num, trial_len_arr):
    # (152730, 62, 5) (152730,) 15subjects 3sessions 3394seconds, 62 channels, 5 DE features
    # [235, 233, 206, 238, 185, 195, 237, 216, 265, 237, 235, 233, 235, 238, 206]
    trial_len_arr = [235, 233, 206, 238, 185, 195, 237, 216, 265, 237, 235, 233, 235, 238, 206] # in seconds

    def moving_average(X, w):
        if len(X.shape) == 1:
            return np.convolve(X, np.ones(w), 'valid') / w
        elif len(X.shape) == 2:
            out = []
            for i in range(X.shape[1]):
                data = np.convolve(X[:, i], np.ones(w), 'valid') / w
                #print(data.shape)
                out.append(data)
            return np.transpose(np.stack(out, axis=0), (1, 0))
        else:
            print('Moving Average ERROR!')
            return None

    data_smoothed = []
    for subject in range(subject_num):
        for session in range(session_num):
            for trial in range(trial_num):
                chunk_num = trial_len_arr[trial]
                start = subject * (X.shape[0] // (subject_num)) + session * \
                        (X.shape[0] // (subject_num * session_num)) + int(np.sum(trial_len_arr[:trial]))
                #print(start, start+chunk_num)
                pre_ma = X[start:start+chunk_num, :]
                after_ma = moving_average(pre_ma, ma_window)
                data_smoothed.append(after_ma)
    data_smoothed = np.concatenate(data_smoothed, axis=0)
    print('feature smooth before/after:', X.shape, data_smoothed.shape)

    return data_smoothed


def process_seizure_data(dataset_name):
    if dataset_name == 'NICU':
        path = '/mnt/data2/sylyoung/EEG/Seizure/NICU/'
        sample_rate = 500
    elif dataset_name == 'CHSZ':
        path = '/mnt/data2/sylyoung/EEG/Seizure/CHSZ/'
        sample_rate = 256

    def sort_func(name_string):
        if 'DS_Store' in name_string:
            return -1
        if name_string.endswith('.mat'):
            id_ = name_string[15:-4]
        return int(id_)

    file_arr = []
    for subdir, dirs, files in os.walk(path):
        for file in sorted(files, key=sort_func):
            f_path = os.path.join(subdir, file)
            if 'DS_Store' in f_path:
                continue
            file_arr.append(f_path)

    data = []
    labels = []
    for file in file_arr:
        mat = sio.loadmat(file)
        print(file)
        X = np.array(mat['X'])
        X = np.transpose(X, (0, 2, 1))
        y = np.array(mat['y']).reshape(-1, )

        if X.shape[-1] == 4000 and dataset_name == 'CHSZ':
            print('downsample...')
            X = mne.filter.resample(X, down=2)

        print(X.shape, y.shape)
        data.append(X)
        labels.append(y)

    return data, labels, sample_rate


def test():
    X = np.load('./data/' + 'SEED' + '/X_DE_session1.npy')
    labels = np.load('./data/' + 'SEED' + '/labels_session1.npy')
    kf = KalmanFilter(n_dim_state=5, n_dim_obs=310)
    #kf = KalmanFilter(em_vars=['transition_covariance', 'observation_covariance'])
    kf.em(X, n_iter=5)
    #print(X[0])
    (filtered_state_means, filtered_state_covariances) = kf.filter(X)
    (smoothed_state_means, smoothed_state_covariances) = kf.smooth(X)
    print(filtered_state_means)
    #print(filtered_state_covariances)
    print(smoothed_state_means)
    #print(smoothed_state_covariances)


def dataset_to_file(dataset_name, data_save):
    moabb.set_log_level("ERROR")
    if dataset_name == 'BNCI2014001':
        dataset = BNCI2014001()
        paradigm = MotorImagery(n_classes=4)
        # (5184, 22, 1001) (5184,) 250Hz 9subjects * 4classes * (72+72)trials for 2sessions
    elif dataset_name == 'BNCI2014002':
        dataset = BNCI2014002()
        paradigm = MotorImagery(n_classes=2)
        # (2240, 15, 2561) (2240,) 512Hz 14subjects * 2classes * (50+30)trials * 2sessions(not namely separately)
    elif dataset_name == 'BNCI2014004':
        dataset = BNCI2014004()
        paradigm = MotorImagery(n_classes=2)
        # (6520, 3, 1126) (6520,) 250Hz 9subjects * 2classes * (?)trials * 5sessions
    elif dataset_name == 'BNCI2015001':
        dataset = BNCI2015001()
        paradigm = MotorImagery(n_classes=2)
        # (5600, 13, 2561) (5600,) 512Hz 12subjects * 2 classes * (200 + 200 + (200 for Subj 8/9/10/11)) trials * (2/3)sessions
    elif dataset_name == 'PhysionetMI':
        dataset = PhysionetMI(imagined=True, executed=False)
        paradigm = MotorImagery(n_classes=2)
        #
    elif dataset_name == 'Cho2017':
        dataset = Cho2017()
        paradigm = MotorImagery(n_classes=2)
        #
    elif dataset_name == 'Wang2016':
        dataset = Wang2016()
        paradigm = SSVEP()
    elif dataset_name == 'MI1':
        info = None
        return info
        # (1400, 59, 300) (1400,) 100Hz 7subjects * 2classes * 200trials * 1session
    elif dataset_name == 'BNCI2015004':
        dataset = BNCI2015004()
        paradigm = MotorImagery(n_classes=2)
        # [160, 160, 160, 150 (80+70), 160, 160, 150 (80+70), 160, 160]
        # (1420, 30, 1793) (1420,) 256Hz 9subjects * 2classes * (80+80/70)trials * 2sessions
    elif dataset_name == 'BNCI2014008':
        dataset = BNCI2014008()
        paradigm = P300()
        # (33600, 8, 257) (33600,) 256Hz 8subjects 4200 trials * 1session
    elif dataset_name == 'BNCI2014009':
        dataset = BNCI2014009()
        paradigm = P300()
        # (17280, 16, 206) (17280,) 256Hz 10subjects 1728 trials * 3sessions
    elif dataset_name == 'BNCI2015003':
        dataset = BNCI2015003()
        paradigm = P300()
        # (25200, 8, 206) (25200,) 256Hz 10subjects 2520 trials * 1session
    elif dataset_name == 'EPFLP300':
        dataset = EPFLP300()
        paradigm = P300()
        # (25200, 8, 206) (25200,) 256Hz 10subjects 1session
    elif dataset_name == 'ERN':
        ch_names = ['Fp1', 'Fp2', 'AF7', 'AF3', 'AF4', 'AF8', 'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8',
                    'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'Cz', 'C2',
                    'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3',
                    'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'POz', 'P08', 'O1', 'O2']
        info = mne.create_info(ch_names=ch_names, sfreq=200, ch_types=['eeg'] * 56)
        return info
        # (5440, 56, 260) (5440,) 200Hz 16subjects 1session
    # SEED (152730, 62, 5*DE*)  (152730,) 200Hz 15subjects 3sessions

    if data_save:
        print('preparing data...')
        # dataset.subject_list[:5] or [dataset.subject_list[0]]
        # PhysionetMI 87,91,99 with different time_samples; 103 with different num_trials
        if dataset_name == 'PhysionetMI':
            #print(type(dataset.subject_list[:]))
            #print(list(type(np.delete(dataset.subject_list, [87,91,99,103])))
            X, labels, meta = paradigm.get_data(dataset=dataset, subjects=list(np.delete(dataset.subject_list, [87,91,99,103,104])))
        else:
            X, labels, meta = paradigm.get_data(dataset=dataset, subjects=dataset.subject_list[:])
        ar_unique, cnts = np.unique(labels, return_counts=True)
        print("labels:", ar_unique)
        print("Counts:", cnts)
        print(X.shape, labels.shape)
        np.save('./data/' + dataset_name + '/X', X)
        np.save('./data/' + dataset_name + '/labels', labels)
        meta.to_csv('./data/' + dataset_name + '/meta.csv')
    else:
        if isinstance(paradigm, MotorImagery):
            X, labels, meta = paradigm.get_data(dataset=dataset, subjects=[dataset.subject_list[0]], return_epochs=True)
            return X.info
        elif isinstance(paradigm, P300):
            X, labels, meta = paradigm.get_data(dataset=dataset, subjects=[dataset.subject_list[0]], return_epochs=True)
            return X.info


if __name__ == '__main__':
    #dataset_name = 'BNCI2014001'
    #dataset_name = 'BNCI2014002'
    #dataset_name = 'BNCI2014004'
    #dataset_name = 'BNCI2015001'
    #dataset_name = 'PhysionetMI'
    #dataset_name = 'MI1'
    #dataset_name = 'BNCI2015004'
    #dataset_name = 'BNCI2014008'
    #dataset_name = 'BNCI2014009'
    #dataset_name = 'BNCI2015003'
    #dataset_name = 'EPFLP300'
    #dataset_name = 'Cho2017'
    dataset_name = 'Wang2016'
    info = dataset_to_file(dataset_name, data_save=True)
    print(dataset_name)
    print(info)

    #process_seizure_data('CHSZ')

    #data_folder = '/Users/Riccardo/Workspace/HUST-BCI/data/ERN'
    #dataset_ERN_to_file(data_folder)
    '''
    data_folder = '/Users/Riccardo/Workspace/HUST-BCI/data/SEED/Preprocessed_EEG'
    #data_folder = '/mnt/data2/sylyoung/EEG/SEED/Preprocessed_EEG/'
    dataset_SEED_DE_to_file(data_folder, use_EA=True)
    dataset_SEED_DE_to_file(data_folder, use_EA=False)
    '''
    #data_folder = '/Users/Riccardo/Workspace/HUST-BCI/data/DEAP/data_preprocessed_matlab'
    #data_folder = '/mnt/data2/sylyoung/EEG/SEED/Preprocessed_EEG/'
    #dataset_DEAP_DE_to_file(data_folder, use_EA=True)
    #dataset_DEAP_DE_to_file(data_folder, use_EA=False)

    #data_folder = '/mnt/data2/sylyoung/EEG/SEED-V/SEED-V/EEG_raw/'
    #dataset_SEEDV_DE_to_file(data_folder, use_EA=True)
    #dataset_SEEDV_DE_to_file(data_folder, use_EA=False)

    #data_folder = '/Users/Riccardo/Workspace/HUST-BCI/data/SEED/SEED-V/EEG_DE_features'
    #dataset_SEEDV_extracted_process(data_folder=data_folder)

    #dataset_SEEDV_eye_extracted_process(data_folder='/Users/Riccardo/Workspace/HUST-BCI/data/SEED/SEED-V/Eye_movement_features')

    #feature_smooth_moving_average('/Users/Riccardo/Workspace/HUST-BCI/repos/EEG/data/SEED/X_DE.npy', 5)
    #feature_smooth_moving_average('/Users/Riccardo/Workspace/HUST-BCI/repos/EEG/data/SEED/X_DE_EA.npy')

    #test()

    #dataset_SEEDV_extracted_process('/Users/Riccardo/Workspace/HUST-BCI/data/SEED/SEED-V/EEG_DE_features')
    #dataset_SEEDV_extracted_process('/Users/Riccardo/Workspace/HUST-BCI/data/SEED/SEED-V/Eye_movement_features')


    '''
    BNCI2014001
    <Info | 8 non-empty values
     bads: []
     ch_names: Fz, FC3, FC1, FCz, FC2, FC4, C5, C3, C1, Cz, C2, C4, C6, CP3, ...
     chs: 22 EEG
     custom_ref_applied: False
     dig: 25 items (3 Cardinal, 22 EEG)
     highpass: 8.0 Hz
     lowpass: 32.0 Hz
     meas_date: unspecified
     nchan: 22
     projs: []
     sfreq: 250.0 Hz
    >
    
    BNCI2014002
    <Info | 7 non-empty values
     bads: []
     ch_names: EEG1, EEG2, EEG3, EEG4, EEG5, EEG6, EEG7, EEG8, EEG9, EEG10, ...
     chs: 15 EEG
     custom_ref_applied: False
     highpass: 8.0 Hz
     lowpass: 32.0 Hz
     meas_date: unspecified
     nchan: 15
     projs: []
     sfreq: 512.0 Hz
    >
    
    BNCI2014004
    <Info | 8 non-empty values
     bads: []
     ch_names: C3, Cz, C4
     chs: 3 EEG
     custom_ref_applied: False
     dig: 6 items (3 Cardinal, 3 EEG)
     highpass: 8.0 Hz
     lowpass: 32.0 Hz
     meas_date: unspecified
     nchan: 3
     projs: []
     sfreq: 250.0 Hz
    >
    
    BNCI2015001
    <Info | 8 non-empty values
     bads: []
     ch_names: FC3, FCz, FC4, C5, C3, C1, Cz, C2, C4, C6, CP3, CPz, CP4
     chs: 13 EEG
     custom_ref_applied: False
     dig: 16 items (3 Cardinal, 13 EEG)
     highpass: 8.0 Hz
     lowpass: 32.0 Hz
     meas_date: unspecified
     nchan: 13
     projs: []
     sfreq: 512.0 Hz
    >
    
    PhysionetMI
    <Info | 8 non-empty values
     bads: []
     ch_names: FC5, FC3, FC1, FCz, FC2, FC4, FC6, C5, C3, C1, Cz, C2, C4, C6, ...
     chs: 64 EEG
     custom_ref_applied: False
     dig: 67 items (3 Cardinal, 64 EEG)
     highpass: 8.0 Hz
     lowpass: 32.0 Hz
     meas_date: 2009-08-12 16:15:00 UTC
     nchan: 64
     projs: []
     sfreq: 160.0 Hz
    >
    
    Cho2017
    <Info | 8 non-empty values
     bads: []
     ch_names: Fp1, AF7, AF3, F1, F3, F5, F7, FT7, FC5, FC3, FC1, C1, C3, C5, ...
     chs: 64 EEG
     custom_ref_applied: False
     dig: 67 items (3 Cardinal, 64 EEG)
     highpass: 8.0 Hz
     lowpass: 32.0 Hz
     meas_date: unspecified
     nchan: 64
     projs: []
     sfreq: 512.0 Hz
    >

    BNCI2015004
    <Info | 8 non-empty values
     bads: []
     ch_names: AFz, F7, F3, Fz, F4, F8, FC3, FCz, FC4, T3, C3, Cz, C4, T4, CP3, ...
     chs: 30 EEG
     custom_ref_applied: False
     dig: 33 items (3 Cardinal, 30 EEG)
     highpass: 8.0 Hz
     lowpass: 32.0 Hz
     meas_date: unspecified
     nchan: 30
     projs: []
     sfreq: 256.0 Hz
    >
    
    BNCI2014008
    <Info | 8 non-empty values
     bads: []
     ch_names: Fz, Cz, Pz, Oz, P3, P4, PO7, PO8
     chs: 8 EEG
     custom_ref_applied: False
     dig: 11 items (3 Cardinal, 8 EEG)
     highpass: 1.0 Hz
     lowpass: 24.0 Hz
     meas_date: unspecified
     nchan: 8
     projs: []
     sfreq: 256.0 Hz
    >
    
    BNCI2014009
    <Info | 8 non-empty values
     bads: []
     ch_names: Fz, Cz, Pz, Oz, P3, P4, PO7, PO8, F3, F4, FCz, C3, C4, CP3, CPz, CP4
     chs: 16 EEG
     custom_ref_applied: False
     dig: 19 items (3 Cardinal, 16 EEG)
     highpass: 1.0 Hz
     lowpass: 24.0 Hz
     meas_date: unspecified
     nchan: 16
     projs: []
     sfreq: 256.0 Hz
    >
    
    BNCI2015003
    <Info | 8 non-empty values
     bads: []
     ch_names: Fz, Cz, P3, Pz, P4, PO7, Oz, PO8
     chs: 8 EEG
     custom_ref_applied: False
     dig: 11 items (3 Cardinal, 8 EEG)
     highpass: 1.0 Hz
     lowpass: 24.0 Hz
     meas_date: unspecified
     nchan: 8
     projs: []
     sfreq: 256.0 Hz
    >

    '''