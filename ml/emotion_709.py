import os
import random

import mne
import numpy as np
import scipy.io as sio
from sklearn import preprocessing
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.svm import SVC
from models.FC import FC
from nn_baseline import nn_fixepoch


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
        for file in sorted(files):
            f_path = os.path.join(subdir, file)
            if 'DS_Store' in f_path or not '.mat' in f_path:
                continue
            print(f_path)
            file_arr.append(f_path)

    subj_num = len(file_arr) // 3
    print('DATA_PATH:', data_folder)
    print('SUBJECT_NUM:', subj_num)

    label_arr = np.array([1, 0, -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 0, 1, -1]) + 1

    data = []
    labels = []
    for i in range(len(file_arr)):
        print(file_arr[i])
        mat = sio.loadmat(file_arr[i])
        subject_data = []
        subject_label = []
        for j in range(15):
            x = np.array(mat['de_LDS' + str(j + 1)])
            y = np.full((x.shape[1], 1), label_arr[j]).reshape(-1, )
            x = np.concatenate((x[:16, :], x[18:20, :], x[21:-1, :]))
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
    data = np.transpose(data, (1, 0, 2))
    print(data.shape, labels.shape)
    data = data.reshape(data.shape[0], -1)
    print(data.shape, labels.shape)

    if not os.path.isdir('./data/'):
        print('creating folder ./data/')
        os.mkdir('./data/')
    np.save('./data/' + '709' + '/X_extracted', data)
    np.save('./data/' + '709' + '/labels_extracted', labels)


def data_loader(dataset, use_one_session=False):
    if dataset == '709':
        # provided extracted feature
        X = np.load('./data/' + dataset + '/X_extracted.npy')
        y = np.load('./data/' + dataset + '/labels_extracted.npy')

    if dataset == '709':
        paradigm = 'Emotion'
        num_subjects = 3

        if use_one_session:
            # only use session 1
            indices = []
            for i in range(num_subjects):
                indices.append(np.arange(3454) + (3454 * i * 3))  # extracted DE feature
            indices = np.concatenate(indices, axis=0)
            X = X[indices]
            y = y[indices]

    le = preprocessing.LabelEncoder()
    y = le.fit_transform(y)
    print('data shape:', X.shape, ' labels shape:', y.shape)
    return X, y, num_subjects, paradigm


def traintest_split_cross_subject(dataset, X, y, num_subjects, test_subject_id):
    data_subjects = np.split(X, indices_or_sections=num_subjects, axis=0)
    labels_subjects = np.split(y, indices_or_sections=num_subjects, axis=0)
    test_x = data_subjects.pop(test_subject_id)
    test_y = labels_subjects.pop(test_subject_id)
    train_x = np.concatenate(data_subjects, axis=0)
    train_y = np.concatenate(labels_subjects, axis=0)
    print('Test subject s' + str(test_subject_id))
    print('Training/Test split:', train_x.shape, test_x.shape)
    return train_x, train_y, test_x, test_y


def traintest_split_within_subject(dataset, X, y, num_subjects, num_sessions, test_subject_id, test_session_id):
    data_subjects = np.split(X, indices_or_sections=num_subjects, axis=0)
    labels_subjects = np.split(y, indices_or_sections=num_subjects, axis=0)
    subject_x = data_subjects[test_subject_id]
    subject_y = labels_subjects[test_subject_id]
    data_sessions = np.split(subject_x, indices_or_sections=num_sessions, axis=0)
    labels_sessions = np.split(subject_y, indices_or_sections=num_sessions, axis=0)
    test_x = data_sessions.pop(test_session_id)
    test_y = labels_sessions.pop(test_session_id)
    train_x = np.concatenate(data_sessions, axis=0)
    train_y = np.concatenate(labels_sessions, axis=0)
    print('Test subject s' + str(test_subject_id), ', Test session ' + str(test_session_id))
    print('Training/Test split:', train_x.shape, test_x.shape)
    return train_x, train_y, test_x, test_y


def apply_zscore(train_x, test_x, num_subjects):
    # z-score WITHIN EACH subject
    # train set split into subjects
    train_z = []
    trial_num = int(train_x.shape[0] / (num_subjects - 1))
    for j in range(num_subjects - 1):
        scaler = preprocessing.StandardScaler()
        train_x_tmp = scaler.fit_transform(train_x[trial_num * j: trial_num * (j + 1), :])
        train_z.append(train_x_tmp)
    train_x = np.concatenate(train_z, axis=0)
    # test subject
    scaler = preprocessing.StandardScaler()
    test_x = scaler.fit_transform(test_x)
    return train_x, test_x


def apply_zscore_multiple_sessions(train_x, test_x, num_subjects, num_sessions):
    # z-score WITHIN EACH session
    # train set split into subjects
    train_z = []
    trial_num = int(train_x.shape[0] / ((num_subjects - 1) * num_sessions))
    for j in range((num_subjects - 1) * num_sessions):
        scaler = preprocessing.StandardScaler()
        train_x_tmp = scaler.fit_transform(train_x[trial_num * j: trial_num * (j + 1), :])
        train_z.append(train_x_tmp)
    train_x = np.concatenate(train_z, axis=0)
    # test subject
    test_z = []
    for j in range(num_sessions):
        scaler = preprocessing.StandardScaler()
        test_x_tmp = scaler.fit_transform(test_x[trial_num * j: trial_num * (j + 1), :])
        test_z.append(test_x_tmp)
    test_x = np.concatenate(test_z, axis=0)
    return train_x, test_x


def apply_zscore_within_sessions(train_x, test_x, num_sessions):
    # z-score WITHIN EACH subject for MULTIPLE session
    # train set split into sessions
    train_z = []
    trial_num = int(train_x.shape[0] / (num_sessions - 1))
    for j in range(num_sessions - 1):
        scaler = preprocessing.StandardScaler()
        train_x_tmp = scaler.fit_transform(train_x[trial_num * j: trial_num * (j + 1), :])
        train_z.append(train_x_tmp)
    train_x = np.concatenate(train_z, axis=0)
    # test session
    scaler = preprocessing.StandardScaler()
    test_x = scaler.fit_transform(test_x)
    return train_x, test_x


def apply_zscore_traintest(train_x, test_x):
    # z-score for training/test sets
    scaler = preprocessing.StandardScaler()
    train_x = scaler.fit_transform(train_x)
    # test subject
    scaler = preprocessing.StandardScaler()
    test_x = scaler.fit_transform(test_x)
    return train_x, test_x


def classifier(approach, train_x, train_y, test_x):
    if approach == 'LDA':
        clf = LinearDiscriminantAnalysis()
    elif approach == 'LR':
        clf = LogisticRegression(max_iter=1000)
    elif approach == 'SVM':
        clf = SVC()
    # clf = LinearDiscriminantAnalysis()
    # clf = SVC()
    # clf = LinearSVC()
    # clf = KNeighborsClassifier()
    clf.fit(train_x, train_y)
    pred = clf.predict(test_x)
    return pred


def eeg_classification_withinsubject(dataset, approach):
    X, y, num_subjects, paradigm = data_loader(dataset)

    num_sessions = 3

    #scores_arr = []
    for i in range(num_subjects):
        #train_x, train_y, test_x, test_y = traintest_split_cross_subject(dataset, X, y, num_subjects, i)
        scores_arr = []
        for j in range(num_sessions):
            train_x, train_y, test_x, test_y = traintest_split_within_subject(dataset, X, y, num_subjects, num_sessions, i, j)

            print('train_x, train_y, test_x, test_y.shape', train_x.shape, train_y.shape, test_x.shape, test_y.shape)

            ar_unique, cnts = np.unique(y, return_counts=True)
            print("labels:", ar_unique)
            print("Counts:", cnts)

            # z-score standardization
            print('applying z-score:', train_x.shape, ' labels shape:', train_y.shape)
            #train_x, test_x = apply_zscore(train_x, test_x, num_subjects)
            #train_x, test_x = apply_zscore_multiple_sessions(train_x, test_x, num_subjects, num_sessions)
            #train_x, test_x = apply_zscore_traintest(train_x, test_x)
            train_x, test_x = apply_zscore_within_sessions(train_x, test_x, num_sessions)

            # Upsampling
            # train_x_xdawn, train_y = apply_smote(train_x_xdawn, train_y)
            # train_x, train_y = apply_randup(train_x, train_y)

            # classifier
            if approach != 'FC':
                pred = classifier(approach, train_x, train_y, test_x)
                score = np.round(balanced_accuracy_score(test_y, pred), 5)
            else:
                feature_in = train_x.shape[1]
                class_out = len(np.unique(y))
                score = nn_fixepoch(model=FC(nn_in=feature_in, nn_out=class_out),
                                    learning_rate=0.0001,
                                    num_iterations=100,
                                    metrics=balanced_accuracy_score,
                                    cuda=False,
                                    cuda_device_id=-1,  # CPU
                                    seed=42,
                                    dataset=dataset,
                                    model_name='FC',
                                    test_subj_id=i,
                                    label_probs=False,
                                    valid_percentage=0,
                                    train_x=train_x,
                                    train_y=train_y,
                                    test_x=test_x,
                                    test_y=test_y)


            print('bca:', score)
            scores_arr.append(score)

        print('#' * 40)
        for i in range(len(scores_arr)):
            scores_arr[i] = np.round(scores_arr[i] * 100)
        print('sbj scores', scores_arr)
        print('avg', np.round(np.average(scores_arr), 5))


def eeg_classification_crosssubject(dataset, approach):
    X, y, num_subjects, paradigm = data_loader(dataset, use_one_session=True)

    num_sessions = 1

    scores_arr = []
    for i in range(num_subjects):
        train_x, train_y, test_x, test_y = traintest_split_cross_subject(dataset, X, y, num_subjects, i)
        #scores_arr = []
        #for j in range(num_sessions):
            #train_x, train_y, test_x, test_y = traintest_split_within_subject(dataset, X, y, num_subjects, num_sessions, i, j)

        print('train_x, train_y, test_x, test_y.shape', train_x.shape, train_y.shape, test_x.shape, test_y.shape)

        ar_unique, cnts = np.unique(y, return_counts=True)
        print("labels:", ar_unique)
        print("Counts:", cnts)

        # z-score standardization
        print('applying z-score:', train_x.shape, ' labels shape:', train_y.shape)
        train_x, test_x = apply_zscore(train_x, test_x, num_subjects)
        #train_x, test_x = apply_zscore_multiple_sessions(train_x, test_x, num_subjects, num_sessions)
        #train_x, test_x = apply_zscore_traintest(train_x, test_x)
        #train_x, test_x = apply_zscore_within_sessions(train_x, test_x, num_sessions)

        # Upsampling
        # train_x_xdawn, train_y = apply_smote(train_x_xdawn, train_y)
        # train_x, train_y = apply_randup(train_x, train_y)

        # classifier
        if approach != 'FC':
            pred = classifier(approach, train_x, train_y, test_x)
            score = np.round(balanced_accuracy_score(test_y, pred), 5)
        else:
            feature_in = train_x.shape[1]
            class_out = len(np.unique(y))
            score = nn_fixepoch(model=FC(nn_in=feature_in, nn_out=class_out),
                                learning_rate=0.0001,
                                num_iterations=100,
                                metrics=balanced_accuracy_score,
                                cuda=False,
                                cuda_device_id=-1,  # CPU
                                seed=42,
                                dataset=dataset,
                                model_name='FC',
                                test_subj_id=i,
                                label_probs=False,
                                valid_percentage=0,
                                train_x=train_x,
                                train_y=train_y,
                                test_x=test_x,
                                test_y=test_y)


        print('bca:', score)
        scores_arr.append(score)

    print('#' * 40)
    for i in range(len(scores_arr)):
        scores_arr[i] = np.round(scores_arr[i] * 100)
    print('sbj scores', scores_arr)
    print('avg', np.round(np.average(scores_arr), 5))




def preprocess_edf(root_path, sample_rate, save_dir):

    video_points = [ [ 22, 262 ], [ 287, 524 ], [ 548, 757 ], [ 780, 1022 ], [ 1047, 1235 ], [ 1258, 1457 ],
                     [ 1481, 1720 ], [ 1745, 1967 ], [ 1990, 2259 ], [ 2281, 2523 ], [ 2548, 2787 ], [ 2810, 3045 ],
                     [ 3069, 3307 ], [ 3330, 3573 ], [ 3596, 3808 ] ]
    labels = [1, 0, -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 0, 1, -1]  # TODO: assmuing same order as original SEED
    #name_arr = ['zj', 'wml', 'lh']
    #time_diff = [ [283, 78, 64], [11, 122, 78], [214, 71, 79]]  # assuming files sorted according to recorded dates
    name_arr = ['lh', 'wml', 'zj']
    time_diff = [[214, 71, 79], [11, 122, 78], [283, 78, 64]]  # assuming files sorted according to recorded dates

    cnts = np.zeros(len(time_diff), dtype=int)

    file_arr = []
    for subdir, dirs, files in os.walk(root_path):
        for file in sorted(files):
            f_path = os.path.join(subdir, file)
            if not '.edf' in f_path:
                continue
            file_arr.append(f_path)

    for file in file_arr:
        print(file)
        data = mne.io.read_raw_edf(file)
        raw_data = data.get_data()
        info = data.info
        channels = data.ch_names
        num_chs, num_timesamples = raw_data.shape

        name = os.path.basename(os.path.dirname(file))
        name_id = name_arr.index(name)
        curr_time_diff = time_diff[name_id]

        # cut from start (time_diff)
        raw_data = raw_data[:, curr_time_diff[cnts[name_id]] * sample_rate:]
        cnts[name_id] += 1

        # cut from end (3808)
        raw_data = raw_data[:, :3808 * sample_rate + 1]

        assert raw_data.shape[1] // 300 == 3808

        mdic = {}

        for i in range(len(video_points)):
            video_clip = raw_data[:, video_points[i][0] * sample_rate - 1:video_points[i][1] * sample_rate]
            x_str = str(name) + '_eeg' + str(i + 1)
            #print(x_str)
            mdic[x_str] = video_clip
            #print(video_clip.shape)

        mdic['label'] = labels

        save_path = save_dir + str(os.path.basename(file)[:-4]) + '.mat'

        sio.savemat(save_path, mdic)


if __name__ == '__main__':

    """
    sample_rate = 300
    csv_path = '/Users/Riccardo/Workspace/HUST-BCI/repos/EEG/data/709/BCI_data'
    save_dir = '/Users/Riccardo/Workspace/HUST-BCI/repos/EEG/data/709/Preprocessed_EEG/'
    preprocess_edf(csv_path, sample_rate, save_dir)
    """
    ###################
    # run matlab codes#
    ###################

    dataset = '709'
    data_folder = '/Users/Riccardo/Workspace/HUST-BCI/repos/EEG/data/709/feature_smooth'
    dataset_SEED_extracted_process(data_folder)

    for approach in ['SVM']:
        print('#' * 40)
        print('#' * 10 + str(approach) + '#' * 10)
        print('#' * 40)

        seed = 42
        random.seed(seed)
        np.random.seed(seed)

        print('#' * 40)
        print('#' * 10 + 'WITHIN SUBJECT' + '#' * 10)
        print('#' * 40)
        eeg_classification_withinsubject(dataset, approach)
        print()
        print('#' * 40)
        print('#' * 10 + 'CROSS SUBJECT' + '#' * 10)
        print('#' * 40)
        eeg_classification_crosssubject(dataset, approach)

