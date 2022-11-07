import os
import random

import numpy as np
import scipy.io as sio
from sklearn import preprocessing
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score


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
    np.save('./data/' + 'SEED' + '/X_extracted', data)
    np.save('./data/' + 'SEED' + '/labels_extracted', labels)


def data_loader(dataset):
    if dataset == 'SEED':
        # provided extracted feature
        X = np.load('./data/' + dataset + '/X_extracted.npy')
        y = np.load('./data/' + dataset + '/labels_extracted.npy')

    if dataset == 'SEED':
        paradigm = 'Emotion'
        num_subjects = 15

        # only use session 1
        indices = []
        for i in range(num_subjects):
            indices.append(np.arange(3394) + (3394 * i * 3))  # extracted DE feature
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


def classifier(approach, train_x, train_y, test_x):
    if approach == 'LDA':
        clf = LinearDiscriminantAnalysis()
    elif approach == 'LR':
        clf = LogisticRegression(max_iter=1000)
    # clf = LinearDiscriminantAnalysis()
    # clf = SVC()
    # clf = LinearSVC()
    # clf = KNeighborsClassifier()
    clf.fit(train_x, train_y)
    pred = clf.predict(test_x)
    return pred


def eeg_classification(dataset, approach):
    X, y, num_subjects, paradigm = data_loader(dataset)

    scores_arr = []
    for i in range(num_subjects):
        train_x, train_y, test_x, test_y = traintest_split_cross_subject(dataset, X, y, num_subjects, i)

        print('train_x, train_y, test_x, test_y.shape', train_x.shape, train_y.shape, test_x.shape, test_y.shape)

        ar_unique, cnts = np.unique(y, return_counts=True)
        print("labels:", ar_unique)
        print("Counts:", cnts)

        # z-score standardization
        print('applying z-score:', train_x.shape, ' labels shape:', train_y.shape)
        train_x, test_x = apply_zscore(train_x, test_x, num_subjects)

        # Upsampling
        # train_x_xdawn, train_y = apply_smote(train_x_xdawn, train_y)
        # train_x, train_y = apply_randup(train_x, train_y)

        # classifier
        pred = classifier(approach, train_x, train_y, test_x)
        score = np.round(balanced_accuracy_score(test_y, pred), 5)

        print('bca:', score)
        scores_arr.append(score)

    print('#' * 40)
    for i in range(len(scores_arr)):
        scores_arr[i] *= 100
    print('sbj scores', scores_arr)
    print('avg', np.round(np.average(scores_arr), 5))


if __name__ == '__main__':
    dataset = 'SEED'
    data_folder = '/Users/Riccardo/Workspace/HUST-BCI/data/SEED/ExtractedFeatures'
    dataset_SEED_extracted_process(data_folder)

    approach = 'LR'

    print(dataset, approach)

    seed = 42
    random.seed(seed)
    np.random.seed(seed)

    eeg_classification(dataset, approach)
