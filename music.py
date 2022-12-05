import random

import mne
import numpy as np
import torch
from mne.preprocessing import Xdawn
from sklearn import preprocessing
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from xgboost import XGBClassifier
from models.CNN import ConvChannelWise
from nn_baseline import nn_fixepoch


def convert_label(labels, axis, threshold):
    # Converting labels to 0 or 1, based on a certain threshold
    label_01 = np.where(labels > threshold, 1, 0)
    #print(label_01)
    return label_01


def data_loader_feature(dataset):
    '''

    :param dataset: str, dataset name
    :return: X, y, num_subjects, paradigm, sample_rate
    '''
    mne.set_log_level('warning')


    X = np.load('./data/' + dataset + '_inter_feature.npz')
    y = np.load('./data/' + dataset + '/y.npy')

    lst = X.files
    X_tmp = []
    for item in lst:
        X_tmp.append(X[item])
    X = np.concatenate(X_tmp, axis=1)

    print(X.shape, y.shape)

    paradigm = 'Music'
    num_subjects = 1
    sample_rate = 250
    ch_num = 8

    import sys
    print(y)
    y = convert_label(y, 0, 0.5)
    print(y)
    sys.exit(0)
    le = preprocessing.LabelEncoder()
    y = le.fit_transform(y)
    print('data shape:', X.shape, ' labels shape:', y.shape)
    return X, y, num_subjects, paradigm, sample_rate, ch_num


def ml_classifier(approach, output_probability, train_x, train_y, test_x, return_model=None):
    if approach == 'LDA':
        clf = LinearDiscriminantAnalysis()
    elif approach == 'LR':
        clf = LogisticRegression(max_iter=1000)
    elif approach == 'AdaBoost':
        clf = AdaBoostClassifier()
    elif approach == 'GradientBoosting':
        clf = GradientBoostingClassifier()
    elif approach == 'xgb':
        clf = XGBClassifier()
    clf.fit(train_x, train_y)

    if output_probability:
        pred = clf.predict_proba(test_x)
    else:
        pred = clf.predict(test_x)
    if return_model:
        return pred, clf
    else:
        return pred


def split_onesubj_traintest(data, labels, percent):
    # TODO: 瞎写的，划分建议重写
    arr = [15, 15, 16, 16, 14, 15, 14, 15, 15, 15, 16, 16, 15, 14, 15, 15, 14, 15, 16, 15, 15, 15, 15, 15, 16, 16, 15, 14, 15, 14, 16, 15]
    assert int(np.sum(arr)) == len(labels), 'Wrong len!'
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    for i in range(len(arr)):
        all_inds = np.arange(int(arr[i]), dtype=int) + int(np.sum(arr[:i]))
        print('train ind / test ind:', all_inds[:int(len(all_inds) * percent)], ';', all_inds[int(len(all_inds) * percent):])

        train_x.append(data[all_inds[int(len(all_inds) * (1 - percent)):]])
        train_y.append(labels[all_inds[int(len(all_inds) * (1 - percent)):]])
        test_x.append(data[all_inds[:int(len(all_inds) * (1 - percent))]])
        test_y.append(labels[all_inds[:int(len(all_inds) * (1 - percent))]])
    train_x = np.concatenate(train_x)
    test_x = np.concatenate(test_x)
    train_y = np.concatenate(train_y)
    test_y = np.concatenate(test_y)

    return train_x, train_y, test_x, test_y


def split_onesubj_traintest_enforce_split(data, labels, percent):

    arr = [15, 15, 16, 16, 14, 15, 14, 15, 15, 15, 16, 16, 15, 14, 15, 15, 14, 15, 16, 15, 15, 15, 15, 15, 16, 16, 15, 14, 15, 14, 16, 15]
    assert int(np.sum(arr)) == len(labels), 'Wrong len!'
    train_x1 = data[:len(data) // 4]
    train_y1 = labels[:len(labels) // 4]
    test_x1 = data[len(data) // 4:len(data) // 2]
    test_y1 = labels[len(labels) // 4:len(data) // 2]
    train_x2 = data[len(data) // 2:(len(data) // 4) * 3]
    train_y2 = labels[len(data) // 2:(len(data) // 4) * 3]
    test_x2 = data[(len(data) // 4) * 3:]
    test_y2 = labels[(len(data) // 4) * 3:]
    train_x = np.concatenate([train_x1, train_x2])
    train_y = np.concatenate([train_y1, train_y2])
    test_x = np.concatenate([test_x1, test_x2])
    test_y = np.concatenate([test_y1, test_y2])
    return train_x, train_y, test_x, test_y


def apply_zscore(train_x, test_x, num_subjects):
    # train split into subjects
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


def eeg_handfeature(dataset,approach):
    X, y, num_subjects, paradigm, sample_rate, ch_num = data_loader_feature(dataset)
    print('X, y, num_subjects, paradigm, sample_rate:', X.shape, y.shape, num_subjects, paradigm, sample_rate)

    train_x, train_y, test_x, test_y = split_onesubj_traintest_enforce_split(X, y, 0.5)

    print('train_x, train_y, test_x, test_y.shape', train_x.shape, train_y.shape, test_x.shape, test_y.shape)

    feature_num = 74

    # mean by channel
    #train_x = np.mean(train_x.reshape(train_x.shape[0], 74, -1), axis=2)
    #test_x = np.mean(test_x.reshape(test_x.shape[0], 74, -1), axis=2)

    #print('After mean by channel: train_x, train_y, test_x, test_y.shape', train_x.shape, train_y.shape, test_x.shape, test_y.shape)

    #train_x = train_x[:, :148]
    #test_x = test_x[:, :148]
    #print('After using only one channel: train_x, train_y, test_x, test_y.shape', train_x.shape, train_y.shape, test_x.shape, test_y.shape)

    '''
    # z-score standardization
    print('applying z-score train_x:', train_x.shape, ' test_x:', test_x.shape)
    scaler = preprocessing.StandardScaler()
    train_x = scaler.fit_transform(train_x)
    scaler = preprocessing.StandardScaler()
    test_x = scaler.fit_transform(test_x)
    '''

    train_x = train_x.reshape(train_x.shape[0], ch_num, feature_num)
    test_x = test_x.reshape(test_x.shape[0], ch_num, feature_num)


    cuda_device_id = '2'
    device = torch.device('cuda:' + cuda_device_id)

    if approach == 'CNN':
        # CNN model
        feature_in = train_x.shape[1]
        class_out = len(np.unique(y))
        # TODO
        model = ConvChannelWise(nn_deep=feature_num,
                                nn_out=class_out,
                                ch_num=ch_num,
                                in_channels=ch_num,
                                out_channels=4,
                                bias=False)
        # model = model.to(torch.device('cuda:0'))
        # summary(model, (ch_num, feature_num))
        score = nn_fixepoch(model=model,
                            learning_rate=0.005,
                            num_iterations=200,
                            metrics=accuracy_score,
                            cuda=True,
                            cuda_device_id=cuda_device_id,
                            seed=42,
                            dataset=dataset,
                            model_name='FC',
                            test_subj_id=-1,
                            label_probs=False,
                            valid_percentage=0,
                            train_x=train_x,
                            train_y=train_y,
                            test_x=test_x,
                            test_y=test_y)


    #pred, model = ml_classifier(approach, False, train_x, train_y, test_x, return_model=True)
    #score = np.round(accuracy_score(test_y, pred), 5)

    #print('score', np.round(score, 5))

    return score


if __name__ == '__main__':

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    dataset = 'Music'
    approach = 'xgb' # XGBoost
    approach = 'CNN'  #
    eeg_handfeature(dataset, approach)
