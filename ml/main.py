import random
import sys
import os

import braindecode.models
import mne
import numpy as np
import torch
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from matplotlib import pyplot as plt
from mne.decoding import CSP
from mne.preprocessing import Xdawn
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity
from mne.decoding import CSP
from torch.utils.data import TensorDataset, DataLoader
from xgboost import XGBClassifier
from imxgboost.imbalance_xgb import imbalance_xgboost as imb_xgb
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from torchsummary import summary

from EEG.models.Autoencoder import Autoencoder, Autoencoder_encoder
from EEG.models.EEGNet import EEGNet_feature, EEGNet, EEGNetSiameseFusion, EEGNetCNNFusion, EEGNetxy
from EEG.models.EEGWaveNet import EEGWaveNet
#from models.CE_stSENet import CE_stSENet
from EEG.models.FC import FC
from EEG.models.RBM import RBM
from EEG.ml.nn_baseline import nn_fixepoch, nn_fixepoch_siamesefusion, nn_fixepoch_SFN, nn_cotrain, nn_fixepoch_ms
from EEG.utils.alg_utils import EA, LA
from EEG.utils.data_utils import traintest_split_cross_subject, traintest_split_cross_subject_uneven_multiple, dataset_to_file, time_cut, process_seizure_data
from EEG.models.CNN import ConvFeatureChannel, ConvChannelWise
from EEG.models.BEEGNet import BEEGNet



def apply_pca(train_x, test_x, variance_retained):
    pca = PCA(variance_retained)
    print('before PCA:', train_x.shape, test_x.shape)
    train_x = pca.fit_transform(train_x)
    test_x = pca.transform(test_x)
    print('after PCA:', train_x.shape, test_x.shape)
    print('PCA variance retained:', np.sum(pca.explained_variance_ratio_))
    return train_x, test_x


def apply_zscore(train_x, test_x, num_subjects):
    # train split into subjects
    print('applying z-score train_x:', train_x.shape, ' test_x:', test_x.shape)
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


def apply_zscore_multisessions(train_x, test_x, num_subjects, session_num):
    # train split into subjects
    train_z = []
    trial_num = int(train_x.shape[0] / (num_subjects - 1) / session_num)
    for i in range(num_subjects - 1):
        for j in range(session_num):
            scaler = preprocessing.StandardScaler()
            train_x_tmp = scaler.fit_transform(
                train_x[trial_num * i * session_num + trial_num * j: trial_num * i * session_num + trial_num * (j + 1),
                :])
            train_z.append(train_x_tmp)
    train_x = np.concatenate(train_z, axis=0)
    # test subject
    test_z = []
    for j in range(session_num):
        scaler = preprocessing.StandardScaler()
        test_x_tmp = scaler.fit_transform(
            test_x[trial_num * j: trial_num * (j + 1), :])
        test_z.append(test_x_tmp)
    test_x = np.concatenate(test_z, axis=0)
    return train_x, test_x


def apply_smote(train_x, train_y):
    smote = SMOTE(n_jobs=8)
    print('before SMOTE:', train_x.shape, train_y.shape)
    train_x, train_y = smote.fit_resample(train_x, train_y)
    print('after SMOTE:', train_x.shape, train_y.shape)
    return train_x, train_y


def apply_randup(train_x, train_y):
    # TODO subject based random upsampling
    sampler = RandomOverSampler()
    print('before Random Upsampling:', train_x.shape, train_y.shape)
    train_x, train_y = sampler.fit_resample(train_x, train_y)
    print('after Random Upsampling:', train_x.shape, train_y.shape)
    return train_x, train_y


def apply_randdown(train_x, train_y):
    sampler = RandomUnderSampler()
    print('before Random Downsampling:', train_x.shape, train_y.shape)
    train_x, train_y = sampler.fit_resample(train_x, train_y)
    print('after Random Downsampling:', train_x.shape, train_y.shape)
    return train_x, train_y


def similarity_score(X1, X2):
    X1 = X1.reshape(1, -1)
    X2 = X2.reshape(1, -1)
    return cosine_similarity(X1, X2)


def data_loader(dataset):
    '''

    :param dataset: str, dataset name
    :return: X, y, num_subjects, paradigm, sample_rate
    '''
    mne.set_log_level('warning')

    if dataset == 'MI1':
        data = np.load('./data/' + dataset + '/MI1.npz')
        X = data['data']
        X = X.reshape(-1, X.shape[2], X.shape[3])
        y = data['label']
        y = y.reshape(-1, )
    if dataset == 'BNCI2014001-4':
        X = np.load('./data/' + 'BNCI2014001' + '/X.npy')
        y = np.load('./data/' + 'BNCI2014001' + '/labels.npy')
    else:
        X = np.load('./data/' + dataset + '/X.npy')
        y = np.load('./data/' + dataset + '/labels.npy')
    print(X.shape, y.shape)

    num_subjects, paradigm, sample_rate = None, None, None

    if dataset == 'BNCI2014001':
        paradigm = 'MI'
        num_subjects = 9
        sample_rate = 250
        ch_num = 22

        # only use session T, remove session E
        indices = []
        for i in range(num_subjects):
            indices.append(np.arange(288) + (576 * i))
        indices = np.concatenate(indices, axis=0)
        X = X[indices]
        y = y[indices]

        # only use two classes [left_hand, right_hand]
        indices = []
        for i in range(len(y)):
            if y[i] in ['left_hand', 'right_hand']:
                indices.append(i)
        X = X[indices]
        y = y[indices]
    elif dataset == 'BNCI2014001-4':
        paradigm = 'MI'
        num_subjects = 9
        sample_rate = 250
        ch_num = 22

        # only use session T, remove session E
        indices = []
        for i in range(num_subjects):
            indices.append(np.arange(288) + (576 * i))
        indices = np.concatenate(indices, axis=0)
        X = X[indices]
        y = y[indices]
    elif dataset == 'BNCI2014002':
        paradigm = 'MI'
        num_subjects = 14
        sample_rate = 512
        ch_num = 15

        # only use session train, remove session test
        indices = []
        for i in range(num_subjects):
            indices.append(np.arange(100) + (160 * i))
        indices = np.concatenate(indices, axis=0)
        X = X[indices]
        y = y[indices]

    elif dataset == 'BNCI2015001':
        paradigm = 'MI'
        num_subjects = 12
        sample_rate = 512
        ch_num = 13

        # only use session 1, remove session 2/3
        indices = []
        for i in range(num_subjects):
            if i in [7, 8, 9, 10]:
                indices.append(np.arange(200) + (400 * 7) + 600 * (i - 7))
            elif i == 11:
                indices.append(np.arange(200) + (400 * 7) + 600 * (i - 7))
            else:
                indices.append(np.arange(200) + (400 * i))
        indices = np.concatenate(indices, axis=0)

        X = X[indices]
        y = y[indices]
    elif dataset == 'BNCI2014004':
        paradigm = 'MI'
        num_subjects = 9
        sample_rate = 250
        ch_num = 3

        trials_arr = np.array([[120, 120, 160, 160, 160],
                    [120, 120, 160, 120, 160],
                    [120, 120, 160, 160, 160],
                    [120, 140, 160, 160, 160],
                    [120, 140, 160, 160, 160],
                    [120, 120, 160, 160, 160],
                    [120, 120, 160, 160, 160],
                    [160, 120, 160, 160, 160],
                    [120, 120, 160, 160, 160]])

        # only use session 1-2, remove session 3-5
        indices = []
        for i in range(num_subjects):
            indices.append(np.arange(np.sum(trials_arr[i, :2])) + np.sum(trials_arr[:i, :]))
        indices = np.concatenate(indices, axis=0)
        X = X[indices]
        y = y[indices]
    elif dataset == 'Cho2017':
        paradigm = 'MI'
        num_subjects = 49
        sample_rate = 512
        ch_num = 64

        trials_arr = np.array([200, 200, 200, 200, 200, 200, 240, 200, 240, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200])

        # remove extra sessions from the two subjects to have identical trial nums
        indices = []
        cnt_extra = 0
        for i in range(num_subjects):
            if i == 6 or i == 8:
                indices.append(np.arange(200) + 200 * i + cnt_extra * 40)
                cnt_extra += 1
            else:
                indices.append(np.arange(200) + 200 * i + cnt_extra * 40)
        indices = np.concatenate(indices, axis=0)
        X = X[indices]
        y = y[indices]

    elif dataset == 'MI1':
        paradigm = 'MI'
        num_subjects = 7
        sample_rate = 100
        ch_num = 59
    elif dataset == 'BNCI2014008':
        paradigm = 'ERP'
        num_subjects = 8
        sample_rate = 256
        ch_num = 8

        # time cut
        X = time_cut(X, cut_percentage=0.8)
    elif dataset == 'BNCI2014009':
        paradigm = 'ERP'
        num_subjects = 10
        sample_rate = 256
        ch_num = 16
    elif dataset == 'BNCI2015003':
        paradigm = 'ERP'
        num_subjects = 10
        sample_rate = 256
        ch_num = 8
    elif dataset == 'PhysionetMI':
        paradigm = 'MI'
        num_subjects = 104
        sample_rate = 160
        ch_num = 64
        '''
        run_cnters = []
        with open('./data/PhysionetMI/meta.csv', 'r') as f:
            subj_id_memo = -1
            run_id_memo = -1
            run_lengths = []
            run_cnter = 0
            for line in f:
                if 'subject' in line:
                    continue
                id_, subj_id, _, run_id = line.replace('\n', '').split(',')
                id_, subj_id, run_id = int(id_), int(subj_id) - 1, int(run_id[-1:])
                if subj_id_memo != subj_id:
                    if run_id_memo != -1:
                        run_lengths.append(run_cnter)
                        run_cnters.append(run_lengths)
                    subj_id_memo = subj_id
                    run_id_memo = run_id
                    run_lengths = []
                    run_cnter = 1
                elif run_id_memo != run_id:
                    run_id_memo = run_id
                    run_lengths.append(run_cnter)
                    run_cnter = 1
                else:
                    run_cnter += 1
                if id_ == 4767:
                    run_lengths.append(run_cnter)
                    run_cnters.append(run_lengths)
        print(run_cnters)
        print(len(run_cnters))
        print(np.sum(run_cnters))
        '''
    le = preprocessing.LabelEncoder()
    y = le.fit_transform(y)
    print('data shape:', X.shape, ' labels shape:', y.shape)
    return X, y, num_subjects, paradigm, sample_rate, ch_num


def data_loader_feature(dataset):
    '''

    :param dataset: str, dataset name
    :return: X, y, num_subjects, paradigm, sample_rate
    '''
    mne.set_log_level('warning')

    X = np.load('./data/' + dataset + '_inter_feature_EA\'d.npz')
    y = np.load('./data/' + dataset + '/labels.npy')

    lst = X.files
    X_tmp = []
    for item in lst:
        X_tmp.append(X[item])
    X = np.concatenate(X_tmp, axis=1)

    print(X.shape, y.shape)
    num_subjects, paradigm, sample_rate = None, None, None

    if dataset == 'BNCI2014001':
        paradigm = 'MI'
        num_subjects = 9
        sample_rate = 250
        ch_num = 22

        # only use session T, remove session E
        indices = []
        for i in range(num_subjects):
            indices.append(np.arange(288) + (576 * i))
        indices = np.concatenate(indices, axis=0)
        X = X[indices]
        y = y[indices]

        # only use two classes [left_hand, right_hand]
        indices = []
        for i in range(len(y)):
            if y[i] in ['left_hand', 'right_hand']:
                indices.append(i)
        X = X[indices]
        y = y[indices]
    elif dataset == 'BNCI2014002':
        paradigm = 'MI'
        num_subjects = 14
        sample_rate = 512
        ch_num = 15

        # only use session train, remove session test
        indices = []
        for i in range(num_subjects):
            indices.append(np.arange(100) + (160 * i))
        indices = np.concatenate(indices, axis=0)
        X = X[indices]
        y = y[indices]

    elif dataset == 'BNCI2015001':
        paradigm = 'MI'
        num_subjects = 12
        sample_rate = 512
        ch_num = 13

        # only use session 1, remove session 2/3
        indices = []
        for i in range(num_subjects):
            if i in [7, 8, 9, 10]:
                indices.append(np.arange(200) + (400 * 7) + 600 * (i - 7))
            elif i == 11:
                indices.append(np.arange(200) + (400 * 7) + 600 * (i - 7))
            else:
                indices.append(np.arange(200) + (400 * i))
        indices = np.concatenate(indices, axis=0)
        X = X[indices]
        y = y[indices]
    elif dataset == 'BNCI2014008':
        paradigm = 'ERP'
        num_subjects = 8
        sample_rate = 256
        ch_num = 8

        # time cut
        # X = time_cut(X, cut_percentage=0.8)
    elif dataset == 'BNCI2014009':
        paradigm = 'ERP'
        num_subjects = 10
        sample_rate = 256
        ch_num = 16
    elif dataset == 'BNCI2015003':
        paradigm = 'ERP'
        num_subjects = 10
        sample_rate = 256
        ch_num = 8

    le = preprocessing.LabelEncoder()
    y = le.fit_transform(y)
    print('data shape:', X.shape, ' labels shape:', y.shape)
    return X, y, num_subjects, paradigm, sample_rate, ch_num


def data_alignment(X, num_subjects):
    '''
    :param X: np array, EEG data
    :param num_subjects: int, number of total subjects in X
    :return: np array, aligned EEG data
    '''
    # subject-wise EA
    print('before EA:', X.shape)
    out = []
    for i in range(num_subjects):
        tmp_x = EA(X[X.shape[0] // num_subjects * i:X.shape[0] // num_subjects * (i + 1), :, :])
        out.append(tmp_x)
    X = np.concatenate(out, axis=0)
    print('after EA:', X.shape)
    return X


def data_alignment_two_sessions(X, num_subjects, session_split):
    '''
    :param X: np array, EEG data of two sessions
    :param num_subjects: int, number of total subjects in X
    :param session_split: float, session split ratio
    :return: np array, aligned EEG data
    '''
    # subject-wise EA
    print('before EA:', X.shape)
    out = []
    for i in range(num_subjects):
        subj_trial_num = int(X.shape[0] // num_subjects)
        first_session_num = int(subj_trial_num * session_split)
        # print(first_session_num, subj_trial_num)
        tmp_x = EA(X[subj_trial_num * i:subj_trial_num * i + first_session_num, :, :])
        out.append(tmp_x)
        tmp_x = EA(X[subj_trial_num * i + first_session_num:subj_trial_num * (i + 1), :, :])
        out.append(tmp_x)
    X = np.concatenate(out, axis=0)
    print('after EA:', X.shape)
    return X


def data_alignment_unequal(X, num_subjects, trial_num_arr):
    '''
    :param X: np array, EEG data
    :param num_subjects: int, number of total subjects in X
    :return: np array, aligned EEG data
    '''
    # subject-wise EA
    print('before EA:', X.shape)
    out = []
    for i in range(num_subjects):
        tmp_x = EA(X[int(np.sum(trial_num_arr[:i])):np.sum(trial_num_arr[:i+1]), :, :])
        out.append(tmp_x)
    X = np.concatenate(out, axis=0)
    print('after EA:', X.shape)
    return X


def data_alignment_arrays(X):
    '''
    :param X: array of np arrays, EEG data
    :return: array of np arrays, aligned EEG data
    '''
    # subject-wise EA
    out = []
    for i in range(len(X)):
        print(X[i].shape)
        tmp_x = EA(X[i])
        print(tmp_x.shape)
        out.append(tmp_x)
    return out


def traintest_split_within_subject(dataset, X, y, num_subjects, test_subject_id, num, shuffle, use_EA):
    print('Within subject s' + str(test_subject_id))
    data_subjects = np.split(X, indices_or_sections=num_subjects, axis=0)
    labels_subjects = np.split(y, indices_or_sections=num_subjects, axis=0)
    subj_data = data_subjects.pop(test_subject_id)
    subj_label = labels_subjects.pop(test_subject_id)
    print('Subject data:', subj_data.shape, subj_label.shape)
    class_out = len(np.unique(subj_label))
    if shuffle:
        inds = np.arange(len(subj_data))
        np.random.shuffle(inds)
        subj_data = subj_data[inds]
        subj_label = subj_label[inds]
    if num < 1:  # percentage
        num_ints = []
        for i in range(class_out):
            num_ints.append(int(len(np.where(subj_label == i)[0])))
        print('num_ints per class:', num_ints)
    else:  # numbers
        num_int = int(num)

    inds_all_train = []
    inds_all_test = []
    for class_num in range(class_out):
        num_int = num_ints[class_num]
        inds_class = np.where(subj_label == class_num)[0]
        print('inds_class len:', len(inds_class))
        inds_all_train.append(inds_class[:int(num_int * num)])
        inds_all_test.append(inds_class[int(num_int * num):])
    inds_all_train = np.concatenate(inds_all_train)
    inds_all_test = np.concatenate(inds_all_test)
    np.set_printoptions(threshold=sys.maxsize)
    train_x = subj_data[inds_all_train]
    train_y = subj_label[inds_all_train]
    #print(train_y)
    test_x = subj_data[inds_all_test]
    test_y = subj_label[inds_all_test]
    #print(test_y)
    #input('')

    if use_EA:
        train_x = EA(train_x)
        test_x = EA(test_x)

    print('Training/Test split:', train_x.shape, test_x.shape)
    return train_x, train_y, test_x, test_y


def traintest_split_within_subject_withothersubjects(dataset, X, y, num_subjects, test_subject_id, num, shuffle, use_EA):
    print('Within subject s' + str(test_subject_id))
    data_subjects = np.split(X, indices_or_sections=num_subjects, axis=0)
    labels_subjects = np.split(y, indices_or_sections=num_subjects, axis=0)
    subj_data = data_subjects.pop(test_subject_id)
    subj_label = labels_subjects.pop(test_subject_id)
    print('Subject data:', subj_data.shape, subj_label.shape)
    class_out = len(np.unique(subj_label))
    if shuffle:
        inds = np.arange(len(subj_data))
        np.random.shuffle(inds)
        subj_data = subj_data[inds]
        subj_label = subj_label[inds]
    if num < 1:  # percentage
        num_ints = []
        for i in range(class_out):
            num_ints.append(int(len(np.where(subj_label == i)[0])))
        print('num_ints per class:', num_ints)
    else:  # numbers
        num_int = int(num)

    other_x_all = []
    other_y_all = []
    for i in range(num_subjects - 1):
        other_x = data_subjects[i]
        other_y = labels_subjects[i]
        if use_EA:
            other_x = EA(other_x)
        other_x_all.append(other_x)
        other_y_all.append(other_y)
    other_x = np.concatenate(other_x_all, axis=0)
    other_y = np.concatenate(other_y_all, axis=0)

    inds_all_train = []
    inds_all_test = []
    for class_num in range(class_out):
        num_int = num_ints[class_num]
        inds_class = np.where(subj_label == class_num)[0]
        print('inds_class len:', len(inds_class))
        inds_all_train.append(inds_class[:int(num_int * num)])
        inds_all_test.append(inds_class[int(num_int * num):])
    inds_all_train = np.concatenate(inds_all_train)
    inds_all_test = np.concatenate(inds_all_test)
    np.set_printoptions(threshold=sys.maxsize)
    train_x = subj_data[inds_all_train]
    train_y = subj_label[inds_all_train]
    test_x = subj_data[inds_all_test]
    test_y = subj_label[inds_all_test]
    if use_EA:
        train_x = EA(train_x)
        test_x = EA(test_x)
    train_x = np.concatenate((train_x, other_x))
    train_y = np.concatenate((train_y, other_y))

    print('Training/Test split:', train_x.shape, test_x.shape)
    return train_x, train_y, test_x, test_y



def ml_classifier(approach, output_probability, train_x, train_y, test_x, return_model=None, weight=None):
    if approach == 'LDA':
        clf = LinearDiscriminantAnalysis()
    elif approach == 'LR':
        clf = LogisticRegression(max_iter=1000)
    elif approach == 'AdaBoost':
        clf = AdaBoostClassifier()
    elif approach == 'GradientBoosting':
        clf = GradientBoostingClassifier()
    elif approach == 'SVM':
        clf = SVC()
    elif approach == 'xgb':
        clf = XGBClassifier()
        if weight:
            print('XGB weight:', weight)
            clf = XGBClassifier(scale_pos_weight=weight)
            # clf = imb_xgb(special_objective='focal', focal_gamma=2.0)
    # clf = LinearDiscriminantAnalysis()
    # clf = SVC()
    # clf = LinearSVC()
    # clf = KNeighborsClassifier()
    clf.fit(train_x, train_y)

    if output_probability:
        pred = clf.predict_proba(test_x)
    else:
        pred = clf.predict(test_x)
    if return_model:
        return pred, clf
    else:
        return pred


def sort_func_gen_data(name_string):
    id_ = -1
    if name_string.endswith('.npy'):
        id_ = int(name_string[-7:-4])
    return id_


def eeg_handfeature(dataset, info, align, approach, cuda_device_id):
    X, y, num_subjects, paradigm, sample_rate, ch_num = data_loader_feature(dataset)

    feature_num = 74
    if paradigm == 'ERP':
        if dataset == 'BNCI2014008' or dataset == 'BNCI2015003':
            print(X.shape, 'before deletion')
            feature_num = 67
            X = np.delete(X, [482, 490, 498, 506, 514, 522, 530, 538], axis=1)
            print(X.shape, 'after deletion')
        else:
            print(X.shape, 'before deletion')
            feature_num = 73
            X = np.delete(X, [1058, 1066, 1074, 1082, 1090, 1098, 1106, 1114, 1122, 1130, 1138, 1146, 1154, 1162, 1170,
                              1178], axis=1)
            print(X.shape, 'after deletion')

    print('X, y, num_subjects, paradigm, sample_rate:', X.shape, y.shape, num_subjects, paradigm, sample_rate)

    if align:
        X = data_alignment(X, num_subjects)

    scores_arr = []

    class_out = len(np.unique(y))

    all_feature_importance = []

    for i in range(num_subjects):
        # train_x, train_y, test_x, test_y = traintest_split_within_subject(dataset, X, y, num_subjects, i, 0.8, True)

        train_x, train_y, test_x, test_y = traintest_split_cross_subject(dataset, X, y, num_subjects, i)

        print('train_x, train_y, test_x, test_y.shape', train_x.shape, train_y.shape, test_x.shape, test_y.shape)

        if paradigm == 'MI':

            ranking = [27, 32, 31, 30, 28, 29, 36, 26, 39, 40, 47, 25, 42, 34, 37, 35, 33, 67, 41, 60, 63, 43, 62, 64
                , 55, 38, 46, 69, 65, 68, 66, 56, 73, 17, 16, 15, 24, 2, 14, 54, 61, 10, 51, 13, 23, 11, 3, 72
                , 57, 7, 6, 49, 9, 1, 20, 8, 53, 18, 59, 22, 71, 12, 48, 21, 0, 4, 19, 52, 45, 58, 70, 50
                , 5, 44]

            ch_num = ch_num

            a = []
            inds_feature = ranking[(-1 * feature_num):]
            # print(inds_feature)
            for k in range(feature_num):
                a.append(np.arange(ch_num, dtype=int) + inds_feature[k] * ch_num)
            a = np.concatenate(a)
            # print(a)
            # input('')

            train_x = train_x[:, a]
            test_x = test_x[:, a]

            # mean by channel
            # train_x = np.mean(train_x.reshape(train_x.shape[0], feature_num, -1), axis=2)
            # test_x = np.mean(test_x.reshape(test_x.shape[0], feature_num, -1), axis=2)

            print('train_x, train_y, test_x, test_y.shape', train_x.shape, train_y.shape, test_x.shape, test_y.shape)

            # train_x_csp = train_x_csp[:, ranking[(-1 * feature_num):]]
            # test_x_csp = test_x_csp[:, ranking[(-1 * feature_num):]]

            # z-score standardization
            print('applying z-score train_x:', train_x.shape, ' test_x:', test_x.shape)
            train_x, test_x = apply_zscore(train_x, test_x, num_subjects)
            # train_x_csp, test_x_csp = apply_zscore_multisessions(train_x_csp, test_x_csp, num_subjects, 2)
            # test_x_csp = test_x_csp[:len(test_x_csp) // 2]
            # test_y = test_y[:len(test_y) // 2]

            # PCA
            # train_x_csp, test_x_csp = apply_pca(train_x_csp, test_x_csp, 0.95)

            train_x = train_x.reshape(train_x.shape[0], ch_num, feature_num)
            test_x = test_x.reshape(test_x.shape[0], ch_num, feature_num)

            if approach == 'CNN':

                # CNN model
                feature_in = train_x.shape[1]
                class_out = len(np.unique(y))
                seed_arr = np.arange(5)
                rand_init_scores = []
                std_arr = []
                for seed in seed_arr:
                    model = ConvChannelWise(nn_deep=feature_num,
                                            nn_out=class_out,
                                            feature_num=feature_num,
                                            in_channels=ch_num,
                                            out_channels=ch_num // 4,
                                            bias=False)
                    rand_init_score = nn_fixepoch(model=model,
                                                  learning_rate=0.001,
                                                  num_iterations=100,
                                                  metrics=accuracy_score,
                                                  cuda=True,
                                                  cuda_device_id=cuda_device_id,
                                                  seed=seed,
                                                  dataset=dataset,
                                                  model_name='FC',
                                                  test_subj_id=i,
                                                  label_probs=False,
                                                  valid_percentage=0,
                                                  train_x=train_x,
                                                  train_y=train_y,
                                                  test_x=test_x,
                                                  test_y=test_y)
                    rand_init_scores.append(rand_init_score)
                print('subj rand_init_scores:', rand_init_scores)
                score = np.round(np.average(rand_init_scores), 5)
                std = np.round(np.std(rand_init_scores), 5)
                std_arr.append(std)

            elif approach != 'FC':
                # classifier
                pred, model = ml_classifier(approach, False, train_x, train_y, test_x, return_model=True)
                # pred = ml_classifier(approach, False, train_x_csp, train_y, test_x_csp)
                # pred = ml_classifier(approach, True, train_x_csp, train_y, train_x_csp)

                # xgboost.plot_importance(model)
                # plt.show()
                # plt.savefig('./figures/' + dataset + '/subject_' + str(i) + '_' + str(feature_num) + '_feature_importance.png')
                # plt.close()

                # mean_feature_importance_by_channel = model.feature_importances_
                # mean_feature_importance_by_channel = np.argsort(mean_feature_importance_by_channel)
                # importance ranking index from low importance to high importance

                # importance_ranking = np.argsort(model.feature_importances_)
                # mean_feature_importance_by_channel = np.mean(model.feature_importances_.reshape(74, -1), axis=1)
                '''
                max_feature_importance_by_channel = np.max(model.feature_importances_.reshape(74, -1), axis=1)
                min_feature_importance_by_channel = np.min(model.feature_importances_.reshape(74, -1), axis=1)
                median_feature_importance_by_channel = np.median(model.feature_importances_.reshape(74, -1), axis=1)
                std_feature_importance_by_channel = np.std(model.feature_importances_.reshape(74, -1), axis=1)

                plt.bar(range(74), mean_feature_importance_by_channel)
                plt.savefig('./figures/' + dataset + '/subject_' + str(i) + '_mean_feature_importance_by_channel.png')
                plt.close()
                plt.bar(range(74), max_feature_importance_by_channel)
                plt.savefig('./figures/' + dataset + '/subject_' + str(i) + '_max_feature_importance_by_channel.png')
                plt.close()
                plt.bar(range(74), min_feature_importance_by_channel)
                plt.savefig('./figures/' + dataset + '/subject_' + str(i) + '_min_feature_importance_by_channel.png')
                plt.close()
                plt.bar(range(74), median_feature_importance_by_channel)
                plt.savefig('./figures/' + dataset + '/subject_' + str(i) + '_median_feature_importance_by_channel.png')
                plt.close()
                plt.bar(range(74), std_feature_importance_by_channel)
                plt.savefig('./figures/' + dataset + '/subject_' + str(i) + '_std_feature_importance_by_channel.png')
                plt.close()
                '''
                # print('mean_feature_importance_by_channel:', mean_feature_importance_by_channel)
                # print('importance_ranking:', importance_ranking)
                # if i != 1 and i != 4:
                #    all_feature_importance.append(importance_ranking)

                # clf = LinearDiscriminantAnalysis()
                # clf.fit(train_x_csp, train_y)
                '''
                pred = clf.predict_proba(train_x_csp)
                score = np.round(accuracy_score(train_y, np.argmax(pred, axis=1)), 5)
                print('testscore', np.round(accuracy_score(test_y, np.argmax(clf.predict_proba(test_x_csp), axis=1)), 5))

                np.save('./files/' + dataset + '_csp_pred_testsubj_' + str(i) + '_.npy', pred)
                np.save('./files/' + dataset + '_csp_pred_classification_testsubj_' + str(i) + '_.npy', np.argmax(pred, axis=1) == train_y)
                '''
                # pred = clf.predict_proba(test_x_csp)
                # score = np.round(accuracy_score(test_y, np.argmax(pred, axis=1)), 5)
                score = np.round(accuracy_score(test_y, pred), 5)
            else:
                # FC model
                feature_in = train_x.shape[1]
                class_out = len(np.unique(y))
                score = nn_fixepoch(model=FC(nn_in=feature_in, nn_out=class_out),
                                    learning_rate=0.0001,
                                    num_iterations=100,
                                    metrics=accuracy_score,
                                    cuda=True,
                                    cuda_device_id=cuda_device_id,
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
            print('acc:', score)
        elif paradigm == 'ERP':

            loss_weights = []
            ar_unique, cnts_class = np.unique(y, return_counts=True)
            print("labels:", ar_unique)
            print("Counts:", cnts_class)
            loss_weights.append(1.0)
            loss_weights.append(cnts_class[0] / cnts_class[1])
            # loss_weights = cnts_class[0] / cnts_class[1]
            # loss_weights = None
            print(loss_weights)
            loss_weights = torch.Tensor(loss_weights)
            loss_weights = loss_weights.to(torch.device('cuda:' + str(cuda_device_id)))

            '''
            ranking = [27, 32, 31, 30, 28, 29, 36, 26, 39, 40, 47, 25, 42, 34, 37, 35, 33, 67, 41, 60, 63, 43, 62, 64
                , 55, 38, 46, 69, 65, 68, 66, 56, 73, 17, 16, 15, 24, 2, 14, 54, 61, 10, 51, 13, 23, 11, 3, 72
                , 57, 7, 6, 49, 9, 1, 20, 8, 53, 18, 59, 22, 71, 12, 48, 21, 0, 4, 19, 52, 45, 58, 70, 50
                , 5, 44]
                '''
            # np.set_printoptions(threshold=sys.maxsize)
            # print(test_x[:5, :])
            '''
            import math
            for a in range(len(train_x)):
                for b in range(len(train_x[0])):
                    if math.isnan(train_x[a, b]):
                        print(a,b)
            for a in range(len(test_x)):
                for b in range(len(test_x[0])):
                    if math.isnan(test_x[a, b]):
                        print(a,b)
            '''

            ch_num = ch_num

            '''
            a = []
            inds_feature = ranking[(-1 * feature_num):]
            # print(inds_feature)
            for k in range(feature_num):
                a.append(np.arange(ch_num, dtype=int) + inds_feature[k] * ch_num)
            a = np.concatenate(a)
            # print(a)
            # input('')

            train_x = train_x[:, a]
            test_x = test_x[:, a]
            '''
            # mean by channel
            # train_x = np.mean(train_x.reshape(train_x.shape[0], feature_num, -1), axis=2)
            # test_x = np.mean(test_x.reshape(test_x.shape[0], feature_num, -1), axis=2)

            print('train_x, train_y, test_x, test_y.shape', train_x.shape, train_y.shape, test_x.shape, test_y.shape)

            # train_x_csp = train_x_csp[:, ranking[(-1 * feature_num):]]
            # test_x_csp = test_x_csp[:, ranking[(-1 * feature_num):]]

            # z-score standardization
            print('applying z-score train_x:', train_x.shape, ' test_x:', test_x.shape)
            train_x, test_x = apply_zscore(train_x, test_x, num_subjects)
            # train_x_csp, test_x_csp = apply_zscore_multisessions(train_x_csp, test_x_csp, num_subjects, 2)
            # test_x_csp = test_x_csp[:len(test_x_csp) // 2]
            # test_y = test_y[:len(test_y) // 2]

            # Upsampling
            # train_x_xdawn, train_y = apply_smote(train_x_xdawn, train_y)
            # train_x, train_y = apply_randup(train_x, train_y)

            # train_x, train_y = apply_randup(train_x, train_y)

            train_x = train_x.reshape(train_x.shape[0], ch_num, feature_num)
            test_x = test_x.reshape(test_x.shape[0], ch_num, feature_num)

            print('train_x, train_y, test_x, test_y.shape', train_x.shape, train_y.shape, test_x.shape, test_y.shape)

            if approach == 'CNN':

                # CNN model
                feature_in = train_x.shape[1]
                class_out = len(np.unique(y))
                seed_arr = np.arange(1)
                rand_init_scores = []
                std_arr = []
                for seed in seed_arr:
                    model = ConvChannelWise(nn_deep=feature_num,
                                            nn_out=class_out,
                                            feature_num=feature_num,
                                            in_channels=ch_num,
                                            out_channels=ch_num // 2,
                                            bias=False)
                    rand_init_score = nn_fixepoch(model=model,
                                                  learning_rate=0.001,
                                                  num_iterations=100,
                                                  metrics=accuracy_score,
                                                  cuda=True,
                                                  cuda_device_id=cuda_device_id,
                                                  seed=seed,
                                                  dataset=dataset,
                                                  model_name='FC',
                                                  test_subj_id=i,
                                                  label_probs=False,
                                                  valid_percentage=0,
                                                  train_x=train_x,
                                                  train_y=train_y,
                                                  test_x=test_x,
                                                  test_y=test_y,
                                                  loss_weights=loss_weights)
                    rand_init_scores.append(rand_init_score)
                print('subj rand_init_scores:', rand_init_scores)
                score = np.round(np.average(rand_init_scores), 5)
                std = np.round(np.std(rand_init_scores), 5)
                std_arr.append(std)
            elif approach != 'FC':
                # classifier
                pred = ml_classifier(approach, False, train_x, train_y, test_x, weight=loss_weights)
                score = np.round(balanced_accuracy_score(test_y, pred), 5)
                '''
                #pred = ml_classifier(approach, False, train_x_xdawn, train_y, test_x_xdawn)

                clf = LinearDiscriminantAnalysis()
                #clf.fit(train_x_xdawn_up, train_y_up)

                pred = clf.predict_proba(train_x_xdawn)
                score = np.round(balanced_accuracy_score(train_y, np.argmax(pred, axis=1)), 5)
                print('testscore', np.round(balanced_accuracy_score(test_y, np.argmax(clf.predict_proba(test_x_xdawn), axis=1)), 5))

                np.save('./files/' + dataset + '_xdawn_pred_testsubj_' + str(i) + '_.npy', pred)
                np.save('./files/' + dataset + '_xdawn_pred_classification_testsubj_' + str(i) + '_.npy', np.argmax(pred, axis=1) == train_y)

                #score = np.round(accuracy_score(train_y, np.argmax(pred, axis=1)), 5)
                #score = 0
                '''
            else:
                # FC model
                feature_in = train_x.shape[1]
                class_out = len(np.unique(y))
                score = nn_fixepoch(model=FC(nn_in=feature_in, nn_out=class_out),
                                    learning_rate=0.001,
                                    num_iterations=100,
                                    metrics=balanced_accuracy_score,
                                    cuda=True,
                                    cuda_device_id=cuda_device_id,
                                    seed=42,
                                    dataset=dataset,
                                    model_name='FC',
                                    test_subj_id=i,
                                    label_probs=False,
                                    valid_percentage=0,
                                    train_x=train_x,
                                    train_y=train_y,
                                    test_x=test_x,
                                    test_y=test_y,
                                    loss_weights=loss_weights)
            print('bca:', score)
        scores_arr.append(score)
    print('#' * 30)
    # print('mean importance:', np.mean(all_feature_importance, axis=0))
    # print(np.argsort(np.mean(all_feature_importance, axis=0)))
    # plt.bar(range(74), np.mean(all_feature_importance, axis=0))
    # plt.show()

    '''
    #rank_score = np.zeros(74) # each score corresponds to sum of index rankings
    rank_score = np.zeros(74 * ch_num)  # each score corresponds to sum of index rankings
    for i in range(int(74 * ch_num)):
        for j in range(num_subjects - 2):
            rank_score[i] += all_feature_importance[j].tolist().index(i)
    rank_score = np.mean(rank_score.reshape(74, -1), axis=1)

    out = np.argsort(rank_score)
    print(out)

    # Figure Size
    fig = plt.figure()

    # Horizontal Bar Plot
    plt.bar(np.arange(74, dtype=int), rank_score)

    # Show Plot
    #plt.show()

    #plt.show()
    plt.savefig('./figures/' + dataset + '/final_EA.png')
    plt.close()
    '''

    print('#' * 40)
    for i in range(len(scores_arr)):
        scores_arr[i] *= 100
    print('sbj scores', scores_arr)
    print('avg', np.round(np.average(scores_arr), 5))

    '''
    for i in range(len(std_arr)):
        std_arr[i] *= 100
    print('sbj stds', std_arr)
    print('std_randinit', np.round(np.average(std_arr), 5))
    '''

    return scores_arr


def eeg_ml(dataset, info, align, approach, cuda_device_id, percentage=None):
    X, y, num_subjects, paradigm, sample_rate, ch_num = data_loader(dataset)
    # X, y, num_subjects, paradigm, sample_rate, ch_num = data_loader_feature(dataset)
    print('X, y, num_subjects, paradigm, sample_rate:', X.shape, y.shape, num_subjects, paradigm, sample_rate)

    if paradigm == 'ERP':
        print('ERP downsampled')
        X = mne.filter.resample(X, down=4)
        sample_rate = int(sample_rate // 4)
    print('sample rate:', sample_rate)

    unaligned = X
    if align:
        if dataset == 'BNCI2014004':
            X = data_alignment_unequal(X, num_subjects, trial_num_arr=[240, 240, 240, 260, 260, 240, 240, 280, 240])  # train sessions
            #X = data_alignment_unequal(X, num_subjects, trial_num_arr=[720, 680, 720, 740, 740, 720, 720, 760, 720])  # all sessions
        else:
            X = data_alignment(X, num_subjects)
        '''
        if percentage is not None:
            trial_num_arr = [int(percentage * X.shape[0] // num_subjects)] * num_subjects
            print(trial_num_arr)
            indices = []
            for i in range(num_subjects):
                indices.append(np.arange(int(percentage * X.shape[0] // num_subjects)) + (i * X.shape[0] // num_subjects))
            indices = np.concatenate(indices, axis=0)
            X_percent = X[indices]
            y_percent = y[indices]
            print('X shape with percentage', percentage, X_percent.shape)
            X_percent = data_alignment(X_percent, num_subjects)
        '''

    scores_arr = []

    class_out = len(np.unique(y))

    all_feature_importance = []

    for i in range(num_subjects):
        # within subject
        #train_x, train_y, test_x, test_y = traintest_split_within_subject(dataset, X, y, num_subjects, i, percentage, False)

        if dataset == 'BNCI2014004':
            train_x, train_y, test_x, test_y = traintest_split_cross_subject_uneven_multiple(dataset, X, y, num_subjects, i, trial_num_arr=np.array([[120, 120],
                                                                                                                                                    [120, 120],
                                                                                                                                                    [120, 120],
                                                                                                                                                    [120, 140],
                                                                                                                                                    [120, 140],
                                                                                                                                                    [120, 120],
                                                                                                                                                    [120, 120],
                                                                                                                                                    [160, 120],
                                                                                                                                                    [120, 120]]))  # train sessions
        elif percentage is not None:
            train_x, train_y, _, _ = traintest_split_cross_subject(dataset, X_percent, y_percent, num_subjects, i)
            _, _, test_x, test_y = traintest_split_cross_subject(dataset, X, y, num_subjects, i)
        else:
            train_x, train_y, test_x, test_y = traintest_split_cross_subject(dataset, X, y, num_subjects, i)

        #valid_x = test_x[:20]
        #valid_y = test_y[:20]
        #test_x = test_x[20:]
        #test_y = test_y[20:]
        #train_x, train_y = LA(train_x, train_y, valid_x, valid_y)
        num_trial = len(test_y)
        train_x_subjs, train_y_subjs = [], []
        for k in range(num_subjects - 1):
            train_x_subj, train_y_subj = LA(train_x[num_trial * k:num_trial * (k + 1)], train_y[num_trial * k:num_trial * (k + 1)], test_x, test_y)
            train_x_subjs.append(train_x_subj)
            train_y_subjs.append(train_y_subj)
        train_x = np.concatenate(train_x_subjs)
        train_y = np.concatenate(train_y_subjs)

        '''
        # DeepInversion
        print('retrieving DeepInversion data...')
        inv_data = []
        inv_label = []
        subj_ids = np.arange(num_subjects, dtype=int)
        for subj_id in subj_ids:
            if subj_id == i:
                continue
            for class_id in range(class_out):
                gen_data_dir = '/home/sylyoung/DeepInversion/samples_test/' + dataset + '/'
                gen_data_path = gen_data_dir + 'subject' + str(subj_id) + '_img_s00' + str(class_id) + '_id'

                #print('gen_data_dir, gen_data_path:', gen_data_dir, gen_data_path)

                for subdir, dirs, files in os.walk(gen_data_dir):
                    #print('subdir, dirs, files:', subdir, dirs, files)
                    for file in sorted(files, key=sort_func_gen_data):
                        f_path = os.path.join(subdir, file)
                        if 'DS_Store' in f_path or not '.npy' in f_path or not gen_data_path in f_path:
                            continue
                        #if not 'batch00000' in f_path or not 'batch00001' in f_path:
                        #    continue
                        #print(f_path)

                        gen_data = np.load(f_path)
                        for trial in range(len(train_x)):
                            sim_score = similarity_score(train_x[trial], gen_data)
                            if sim_score > 0.1:
                                print(sim_score)
                            elif sim_score > 0.5:
                                print(f_path, trial)
                        #print('gen data, class_id:', gen_data.shape, class_id)
                        inv_data.append(gen_data)
                        inv_label.append(class_id)
        inv_data = np.stack(inv_data).astype(np.float64)
        inv_label = np.stack(inv_label).astype(np.float64)
        print('inv_data.shape, inv_label.shape:', inv_data.shape, inv_label.shape)

        #inv_data = data_alignment(inv_data, num_subjects - 1)

        train_x = np.concatenate([train_x, inv_data])
        train_y = np.concatenate([train_y, inv_label])

        #print(train_x.dtype, inv_data.dtype)
        #train_x = inv_data.copy()
        #train_y = inv_label.copy()
        '''

        '''
        if dataset == 'BNCI2014001':
            test_x = test_x[:test_x.shape[0] // 2]
            test_y = test_y[:test_y.shape[0] // 2]
        elif dataset == 'BNCI2014002':
            test_x = test_x[:test_x.shape[0] // 1.6]
            test_y = test_y[:test_y.shape[0] // 1.6]
        '''

        print('train_x, train_y, test_x, test_y.shape', train_x.shape, train_y.shape, test_x.shape, test_y.shape)

        if paradigm == 'MI':
            # CSP
            csp = CSP(n_components=10)
            train_x_csp = csp.fit_transform(train_x, train_y)
            test_x_csp = csp.transform(test_x)

            '''
            # training data cleaning with nearest neighbors
            dist_mat = np.zeros((len(train_x_csp), len(train_x_csp)))
            for i1 in range(len(train_x_csp)):
                dist_mat[i1] = euclidean_distances(train_x_csp[i1].reshape(1, -1), train_x_csp)
            k = 3
            marked_inds = []
            modified_labels = []
            # for data removal & edit labels
            for ik in range(len(train_x_csp)):
                neighbor_inds = np.argsort(dist_mat[ik])[1:(k+1)]  # since dist_mat contain dist to itself
                neighbor_labels = train_y[neighbor_inds].tolist()
                if neighbor_labels.count(neighbor_labels[0]) == len(neighbor_labels) and neighbor_labels[0] != train_y[ik]:
                    marked_inds.append(ik)
                    modified_labels.append(neighbor_labels[0])
            '''
            '''
            # GE
            k = 10
            k_prime = 8
            retained_inds = []
            retained_labels = []
            for ik in range(len(train_x_csp)):
                neighbor_inds = np.argsort(dist_mat[ik])[1:(k + 1)]  # since dist_mat contain dist to itself
                neighbor_labels = train_y[neighbor_inds].tolist()
                class_cnt_arr = [neighbor_labels.count(i) for i in range(class_out)]
                if np.max(class_cnt_arr) >= k_prime:
                    retained_inds.append(ik)
                    retained_labels.append(np.argmax(class_cnt_arr))

            train_x = np.array(train_x[retained_inds])
            train_y = np.array(retained_labels)
            '''
            '''
            # data removal
            print('data removing...')
            for ik in range(len(marked_inds) - 1, -1, -1):
                train_x = np.delete(train_x, marked_inds[ik], axis=0)
                train_y = np.delete(train_y, marked_inds[ik], axis=0)
            

            # edit labels
            print('label editing...')
            for ik in range(len(marked_inds) - 1, -1, -1):
                train_y[marked_inds[ik]] = modified_labels[ik]
            '''
            print('train_x, train_y, test_x, test_y.shape', train_x.shape, train_y.shape, test_x.shape, test_y.shape)

            '''
            # refitting CSP
            print('refitting')
            csp = CSP(n_components=10)
            train_x_csp = csp.fit_transform(train_x, train_y)
            '''

            test_x_csp = csp.transform(test_x)
            '''
            # online
            test_x_csp = []
            for l in range(unaligned.shape[0] // num_subjects):
                if l == 0:
                    continue
                elif l == 1:
                    trial_aligned = EA(unaligned[unaligned.shape[0] // num_subjects * i:unaligned.shape[0] // num_subjects * i + l + 1, :, :])[0]
                    test_x_csp.append(csp.transform(trial_aligned.reshape(1, trial_aligned.shape[0], trial_aligned.shape[1])))
                    trial_aligned = EA(unaligned[unaligned.shape[0] // num_subjects * i:unaligned.shape[0] // num_subjects * i + l + 1, :, :])[1]
                    test_x_csp.append(csp.transform(trial_aligned.reshape(1, trial_aligned.shape[0], trial_aligned.shape[1])))
                else:
                    trial_aligned = EA(unaligned[unaligned.shape[0] // num_subjects * i:unaligned.shape[0] // num_subjects * i + l + 1, :, :])[-1]
                    test_x_csp.append(csp.transform(trial_aligned.reshape(1, trial_aligned.shape[0], trial_aligned.shape[1])))
            test_x_csp = np.concatenate(test_x_csp)
            print('Training/Test split after CSP:', train_x_csp.shape, test_x_csp.shape)
            '''
            '''
            kmeans = KMeans(n_clusters=2, random_state=0).fit(test_x_csp)
            print(kmeans.labels_)
            acc_unsup = accuracy_score(kmeans.labels_, test_y)
            print(np.round(max(acc_unsup, 1 - acc_unsup), 3))



            # unsupervised clustering
            for test_id in range(len(test_x_csp)):
                test_sample = test_x_csp[test_id]
                sim_scores = []
                print(test_y[test_id])
                for test_id_comparison in range(len(test_x_csp)):
                    sim_score = cosine_similarity(test_sample.reshape(1, -1), test_x_csp[test_id_comparison].reshape(1, -1))
                    sim_scores.append(round(sim_score[0][0], 3))
                #print(sim_scores)
                sim_scores_sorted = np.argsort(sim_scores)
                print(test_y[sim_scores_sorted[:int(len(sim_scores_sorted) // 2)]])
                print(test_y[sim_scores_sorted[int(len(sim_scores_sorted) // 2):]])
                print(test_y)
                input('')
            sys.exit(0)
            '''

            '''
            ranking = [43, 36, 23, 35, 2, 50, 22, 32, 21, 60, 65, 40, 58, 63, 69, 45, 55, 34, 46, 70, 57, 49, 51, 44,
                       14, 0, 48, 33, 8, 7, 73, 53, 5, 54, 42, 37, 66, 71, 39, 47, 64, 56, 41, 27, 11, 61, 15, 4, 62,
                       30, 38, 13, 52, 67, 31, 6, 28, 9, 18, 25, 72, 59, 20, 1, 10, 24, 68, 3, 26, 12, 19, 16, 17, 29]

            feature_num = 13
            train_x_csp = train_x
            test_x_csp = test_x

            ch_num = 22

            a = []
            inds_feature = ranking[(-1 * feature_num):]
            #print(inds_feature)
            for k in range(feature_num):
                a.append(np.arange(ch_num, dtype=int) + inds_feature[k] * ch_num)
            a = np.concatenate(a)
            #print(a)
            #input('')

            train_x_csp = train_x_csp[:, a]
            test_x_csp = test_x_csp[:, a]
            '''
            # train_x_csp = np.mean(train_x_csp.reshape(train_x_csp.shape[0], 74, -1), axis=2)
            # test_x_csp = np.mean(test_x_csp.reshape(test_x_csp.shape[0], 74, -1), axis=2)

            # train_x_csp = train_x_csp[:, ranking[(-1 * feature_num):]]
            # test_x_csp = test_x_csp[:, ranking[(-1 * feature_num):]]

            # z-score standardization
            #print('applying z-score train_x:', train_x_csp.shape, ' test_x:', test_x_csp.shape)
            #train_x_csp, test_x_csp = apply_zscore(train_x_csp, test_x_csp, num_subjects)
            # train_x_csp, test_x_csp = apply_zscore_multisessions(train_x_csp, test_x_csp, num_subjects, 2)
            # test_x_csp = test_x_csp[:len(test_x_csp) // 2]
            # test_y = test_y[:len(test_y) // 2]

            # PCA
            # train_x_csp, test_x_csp = apply_pca(train_x_csp, test_x_csp, 0.95)

            if approach != 'FC':
                # classifier
                pred, model = ml_classifier(approach, False, train_x_csp, train_y, test_x_csp, return_model=True)
                # pred = ml_classifier(approach, False, train_x_csp, train_y, test_x_csp)
                # pred = ml_classifier(approach, True, train_x_csp, train_y, train_x_csp)

                # xgboost.plot_importance(model)
                # plt.show()
                # plt.savefig('./figures/' + dataset + '/subject_' + str(i) + '_' + str(feature_num) + '_feature_importance.png')
                # plt.close()

                # mean_feature_importance_by_channel = model.feature_importances_
                # mean_feature_importance_by_channel = np.argsort(mean_feature_importance_by_channel)
                # importance ranking index from low importance to high importance

                '''
                mean_feature_importance_by_channel =  np.mean(model.feature_importances_.reshape(74, -1), axis=1)
                max_feature_importance_by_channel = np.max(model.feature_importances_.reshape(74, -1), axis=1)
                min_feature_importance_by_channel = np.min(model.feature_importances_.reshape(74, -1), axis=1)
                median_feature_importance_by_channel = np.median(model.feature_importances_.reshape(74, -1), axis=1)
                std_feature_importance_by_channel = np.std(model.feature_importances_.reshape(74, -1), axis=1)

                plt.bar(range(74), mean_feature_importance_by_channel)
                plt.savefig('./figures/' + dataset + '/subject_' + str(i) + '_mean_feature_importance_by_channel.png')
                plt.close()
                plt.bar(range(74), max_feature_importance_by_channel)
                plt.savefig('./figures/' + dataset + '/subject_' + str(i) + '_max_feature_importance_by_channel.png')
                plt.close()
                plt.bar(range(74), min_feature_importance_by_channel)
                plt.savefig('./figures/' + dataset + '/subject_' + str(i) + '_min_feature_importance_by_channel.png')
                plt.close()
                plt.bar(range(74), median_feature_importance_by_channel)
                plt.savefig('./figures/' + dataset + '/subject_' + str(i) + '_median_feature_importance_by_channel.png')
                plt.close()
                plt.bar(range(74), std_feature_importance_by_channel)
                plt.savefig('./figures/' + dataset + '/subject_' + str(i) + '_std_feature_importance_by_channel.png')
                plt.close()
                '''
                # print('mean_feature_importance_by_channel:', mean_feature_importance_by_channel)
                # if i != 1 and i != 4:
                #    all_feature_importance.append(mean_feature_importance_by_channel)

                # clf = LinearDiscriminantAnalysis()
                # clf.fit(train_x_csp, train_y)
                '''
                pred = clf.predict_proba(train_x_csp)
                score = np.round(accuracy_score(train_y, np.argmax(pred, axis=1)), 5)
                print('testscore', np.round(accuracy_score(test_y, np.argmax(clf.predict_proba(test_x_csp), axis=1)), 5))

                np.save('./files/' + dataset + '_csp_pred_testsubj_' + str(i) + '_.npy', pred)
                np.save('./files/' + dataset + '_csp_pred_classification_testsubj_' + str(i) + '_.npy', np.argmax(pred, axis=1) == train_y)
                '''
                # pred = clf.predict_proba(test_x_csp)
                # score = np.round(accuracy_score(test_y, np.argmax(pred, axis=1)), 5)
                score = np.round(accuracy_score(test_y, pred), 5)
            else:
                # FC model
                feature_in = train_x_csp.shape[1]
                class_out = len(np.unique(y))
                score = nn_fixepoch(model=FC(nn_in=feature_in, nn_out=class_out),
                                    learning_rate=0.0001,
                                    num_iterations=100,
                                    metrics=accuracy_score,
                                    cuda=True,
                                    cuda_device_id=cuda_device_id,
                                    seed=42,
                                    dataset=dataset,
                                    model_name='FC',
                                    test_subj_id=i,
                                    label_probs=False,
                                    valid_percentage=0,
                                    train_x=train_x_csp,
                                    train_y=train_y,
                                    test_x=test_x_csp,
                                    test_y=test_y)
            print('acc:', score)
        elif paradigm == 'ERP':
            # xDAWN
            xdawn = Xdawn(n_components=X.shape[1])  # number of channels
            train_x_epochs = mne.EpochsArray(train_x, info)
            test_x_epochs = mne.EpochsArray(test_x, info)
            train_x_xdawn = xdawn.fit_transform(train_x_epochs)  # unsupervised
            test_x_xdawn = xdawn.transform(test_x_epochs)
            train_x_xdawn = train_x_xdawn.reshape(train_x_xdawn.shape[0], -1)
            test_x_xdawn = test_x_xdawn.reshape(test_x_xdawn.shape[0], -1)
            print('Training/Test split after xDAWN:', train_x_xdawn.shape, test_x_xdawn.shape)

            # z-score standardization
            #train_x_xdawn, test_x_xdawn = apply_zscore(train_x_xdawn, test_x_xdawn, num_subjects)

            # Upsampling
            # train_x_xdawn, train_y = apply_smote(train_x_xdawn, train_y)
            train_x_xdawn, train_y = apply_randup(train_x_xdawn, train_y)

            # PCA
            train_x_xdawn, test_x_xdawn = apply_pca(train_x_xdawn, test_x_xdawn, 0.95)

            #print(train_y)
            #print(test_y)

            #train_x_xdawn_up, train_y_up = apply_randup(train_x_xdawn, train_y)

            if approach != 'FC':
                # classifier
                pred, model = ml_classifier(approach, False, train_x_xdawn, train_y, test_x_xdawn, return_model=True)
                score = np.round(balanced_accuracy_score(test_y, pred), 5)

                '''
                #pred = ml_classifier(approach, False, train_x_xdawn, train_y, test_x_xdawn)

                clf = LinearDiscriminantAnalysis()
                #clf.fit(train_x_xdawn_up, train_y_up)

                pred = clf.predict_proba(train_x_xdawn)
                score = np.round(balanced_accuracy_score(train_y, np.argmax(pred, axis=1)), 5)
                print('testscore', np.round(balanced_accuracy_score(test_y, np.argmax(clf.predict_proba(test_x_xdawn), axis=1)), 5))

                np.save('./files/' + dataset + '_xdawn_pred_testsubj_' + str(i) + '_.npy', pred)
                np.save('./files/' + dataset + '_xdawn_pred_classification_testsubj_' + str(i) + '_.npy', np.argmax(pred, axis=1) == train_y)

                #score = np.round(accuracy_score(train_y, np.argmax(pred, axis=1)), 5)
                #score = 0
                '''
            else:
                # FC model
                feature_in = train_x_xdawn.shape[1]
                class_out = len(np.unique(y))
                score = nn_fixepoch(model=FC(nn_in=feature_in, nn_out=class_out),
                                    learning_rate=0.0001,
                                    num_iterations=50,
                                    metrics=balanced_accuracy_score,
                                    cuda=True,
                                    cuda_device_id=cuda_device_id,
                                    seed=42,
                                    dataset=dataset,
                                    model_name='FC',
                                    test_subj_id=i,
                                    label_probs=False,
                                    valid_percentage=0,
                                    train_x=train_x_xdawn,
                                    train_y=train_y,
                                    test_x=test_x_xdawn,
                                    test_y=test_y)
            print('bca:', score)
        scores_arr.append(score)
    print('#' * 30)
    # print('mean importance:', np.mean(all_feature_importance, axis=0))
    # plt.bar(range(74), np.mean(all_feature_importance, axis=0))
    '''
    rank_score = np.zeros(74) # each score corresponds to sum of index rankings
    for i in range(74):
        for j in range(num_subjects - 2):
            rank_score[i] += all_feature_importance[j].tolist().index(i)
    out = np.argsort(rank_score)
    print(out)

    # Figure Size
    fig = plt.figure()

    # Horizontal Bar Plot
    plt.bar(np.arange(74, dtype=int), rank_score)

    # Show Plot
    #plt.show()

    #plt.show()
    plt.savefig('./figures/' + dataset + '/final.png')
    plt.close()
    '''
    print('#' * 40)
    for i in range(len(scores_arr)):
        scores_arr[i] *= 100
    print('sbj scores', scores_arr)
    print('avg', np.round(np.average(scores_arr), 5))

    return scores_arr


def eeg_dnn(dataset, info, align, approach, cuda_device_id, percentage=None):
    if dataset == 'NICU' or dataset == 'CHSZ':
        X, y, sample_rate = process_seizure_data(dataset)
        paradigm = 'Seizure'
        num_subjects = len(X)
        ch_num = X[0].shape[1]
        eval_metrics = ['accs', 'sens', 'specs', 'aucs', 'f1s', 'bcas']
    else:
        X, y, num_subjects, paradigm, sample_rate, ch_num = data_loader(dataset)
    print('sample rate:', sample_rate)

    unaligned_X = X.copy()

    within = True

    if align:
        if not within:
            if paradigm == 'Seizure':
                X = data_alignment_arrays(X)
            else:
                X = data_alignment(X, num_subjects)

            if percentage is not None:
                trial_num_arr = [int(percentage * X.shape[0] // num_subjects)] * num_subjects
                print(trial_num_arr)
                indices = []
                for i in range(num_subjects):
                    indices.append(np.arange(int(percentage * X.shape[0] // num_subjects)) + (i * X.shape[0] // num_subjects))
                indices = np.concatenate(indices, axis=0)
                X_percent = X[indices]
                y_percent = y[indices]
                print('X shape with percentage', percentage, X_percent.shape)
                X_percent = data_alignment(X_percent, num_subjects)

    scores_arr = []
    scores_arr_all = [[], [], [], [], [], []]
    std_arr = []
    for i in range(num_subjects):

        #train_x, train_y, test_x, test_y = traintest_split_within_subject(dataset, X, y, num_subjects, i, 0.8, shuffle=False, use_EA=align)
        train_x, train_y, test_x, test_y = traintest_split_within_subject_withothersubjects(dataset, X, y, num_subjects, i, 0.5, shuffle=False, use_EA=align)

        #train_x, train_y, test_x, test_y = traintest_split_cross_subject(dataset, X, y, num_subjects, i)
        if dataset == 'NICU' or dataset == 'CHSZ':
            print('num_subjects:', len(X), ', ch_num:', ch_num)
            trial_num_arr = [1747,856,960,913,887,1871,3853,931,1724,1484,1373,2251,996,1426,960,1677,882,1622,1269,1523,1157,1458,2420,839,901,2462,1175,979,1462,974,2837,1224,992,932,988,954,1048,1242,824]
            #trial_num_arr =

        '''
        if percentage is not None:
            train_x, train_y, _, _ = traintest_split_cross_subject(dataset, X_percent, y_percent, num_subjects, i)
            _, _, test_x, test_y = traintest_split_cross_subject(dataset, X, y, num_subjects, i)
        else:
            train_x, train_y, test_x, test_y = traintest_split_cross_subject(dataset, X, y, num_subjects, i, trial_num_arr)
        '''
        # train_x, train_y = test_x, test_y
        print('train_x, train_y, test_x, test_y.shape', train_x.shape, train_y.shape, test_x.shape, test_y.shape)

        if paradigm == 'MI' and approach == 'transform':
            csp = CSP(n_components=ch_num)  # TODO
            csp.fit_transform(train_x, train_y)
            train_x = csp.X_filtered_fitted
            # assert csp.X_filtered_fitted == csp.X_filtered_transformed, 'ERROR CSP Transform!'
            csp.transform(test_x)
            test_x = csp.X_filtered_transformed
            print('Training/Test split after CSP:', train_x.shape, test_x.shape)
        if paradigm == 'ERP' and approach == 'transform':
            xdawn = Xdawn(n_components=X.shape[1])  # number of channels
            train_x_epochs = mne.EpochsArray(train_x, info)
            test_x_epochs = mne.EpochsArray(test_x, info)
            train_x = xdawn.fit_transform(train_x_epochs)  # unsupervised
            test_x = xdawn.transform(test_x_epochs)
            # train_x = train_x_xdawn.reshape(train_x_xdawn.shape[0], -1)
            # test_x = test_x_xdawn.reshape(test_x_xdawn.shape[0], -1)
            print('Training/Test split after xDAWN:', train_x.shape, test_x.shape)

        '''
        train_x_freqs, test_x_freqs = [], []
        for f in range(5):
            train_x_f, train_y, test_x_f, test_y = traintest_split_cross_subject(dataset, X_freqs[f], y, num_subjects, i)
            train_x_freqs.append(train_x_f)
            test_x_freqs.append(test_x_f)
        '''
        class_out = len(np.unique(train_y))

        '''
        # use only portion of training data
        #portion = 0.5
        portion = 0.8
        subj_data_arr = []
        subj_data_arr_y = []
        for s in range(num_subjects - 1):
            subj_data = train_x[(train_x.shape[0] // (num_subjects - 1)) * s:(train_x.shape[0] // (num_subjects - 1)) * (s + 1)]
            subj_data_y = train_y[(train_y.shape[0] // (num_subjects - 1)) * s:(train_y.shape[0] // (num_subjects - 1)) * (s + 1)]
            inds_all = []
            for class_num in range(class_out):
                inds_class = np.where(subj_data_y == class_num)[0]
                inds_class = inds_class[:int(train_x.shape[0] // (num_subjects - 1) // class_out * portion)]
                inds_all.append(inds_class)
            inds_all = np.concatenate(inds_all)
            subj_data = subj_data[inds_all]
            subj_data_y = subj_data_y[inds_all]

            #subj_data = train_x[(train_x.shape[0] // (num_subjects - 1)) * s:(train_x.shape[0] // (num_subjects - 1)) * s + int((train_x.shape[0] // (num_subjects - 1)) * portion)]
            subj_data_arr.append(subj_data)
            #subj_data_y = train_y[(train_y.shape[0] // (num_subjects - 1)) * s:(train_y.shape[0] // (num_subjects - 1)) * s + int((train_y.shape[ 0] // (num_subjects - 1)) * portion)]
            subj_data_arr_y.append(subj_data_y)
        train_x = np.concatenate(subj_data_arr)
        train_y = np.concatenate(subj_data_arr_y)
        print('train_x, train_y, test_x, test_y.shape', train_x.shape, train_y.shape, test_x.shape, test_y.shape)
        '''

        '''
        # using prediction probabilities of traditional approach to replace actual labels to guide NN learning
        # strict rules
        train_y = np.load('./files/' + dataset + '_xdawn_pred_testsubj_' + str(i) + '_.npy')

        cnt_trust = 0
        cnt_throw = 0
        throw_arr = []
        for p in range(len(train_y)):
            if train_y[p, 0] > 0.6:
                train_y[p, 0] = 1
                train_y[p, 1] = 0
                cnt_trust += 1
            elif train_y[p, 1] > 0.6:
                train_y[p, 0] = 0
                train_y[p, 1] = 1
                cnt_trust += 1
            if (train_y[p, 0] > 0.4 and train_y[p, 0] < 0.6) or (train_y[p, 1] > 0.4 and train_y[p, 1] < 0.6):
                train_y[p, 0] = 0.1
                train_y[p, 1] = 0.1
                throw_arr.append(p)
                cnt_throw += 1
            #elif train_y[p, 0] > 0.4 or train_y[p, 1] < 0.6:
            #
            #    cnt_throw += 1
        print('cnt_throw, cnt_trust, train_y.shape:', cnt_throw, cnt_trust, train_y.shape)

        #train_x = np.delete(train_x, throw_arr, axis=0)
        #train_y = np.delete(train_y, throw_arr, axis=0)
        '''

        '''
        # using prediction on training set of traditional approach to guide NN learning
        # suppress cross-entropy loss weights 
        train_y_bool = np.load('./files/' + dataset + '_csp_pred_classification_testsubj_' + str(i) + '_.npy')
        train_y_prob = np.load('./files/' + dataset + '_csp_pred_testsubj_' + str(i) + '_.npy')
        out_y = np.zeros((len(train_y), class_out))
        assert len(train_y_bool) == len(train_y), 'Wrong file train_y_bool shape!'
        for p in range(len(train_y)):
            if train_y_bool[p] == train_y[p]:
                if train_y[p] == 0:
                    # out_y[p, 0] = 0.5
                    out_y[p, 0] = train_y_prob[p, 0] / 2 + 0.5
                else:
                    # out_y[p, 1] = 0.5
                    out_y[p, 1] = train_y_prob[p, 1] / 2 + 0.5
            else:
                if train_y[p] == 0:
                    # out_y[p, 0] = 0.5
                    out_y[p, 0] = 1
                else:
                    # out_y[p, 1] = 0.5
                    out_y[p, 1] = 1
        train_y = out_y
        '''

        '''
        # Retraining: using loss calculated on the training set after model already fitted to eliminate bad data
        fitted_loss = load_and_calculate_loss_from_model(dataset, cuda_device_id, pre_model_arch, i, train_x, train_y)
        fitted_loss = np.array(fitted_loss)

        loss_good_indices = []
        for l in range(len(fitted_loss)):
            if fitted_loss[l] < 0.6931471824645996:
                loss_good_indices.append(l)
        loss_good_indices = np.array(loss_good_indices)

        # unaligned_X_split = np.split(unaligned_X, indices_or_sections=num_subjects, axis=0)
        unaligned_train_x, unaligned_train_y, _, _ = traintest_split_cross_subject(dataset, unaligned_X, y,
                                                                                   num_subjects, i)

        train_x = []
        train_y = []
        for l in range(num_subjects - 1):
            fitted_loss_indices_subject = np.where(loss_good_indices // (len(y) // num_subjects) == l)[0]
            subject_indices = loss_good_indices[fitted_loss_indices_subject]
            unaligned_data_subject = unaligned_train_x[subject_indices]
            aligned_data_subject = data_alignment(unaligned_data_subject, 1)
            train_x.append(aligned_data_subject)
            train_y.append(unaligned_train_y[subject_indices])
        train_x = np.concatenate(train_x)
        train_y = np.concatenate(train_y)
        # np.set_printoptions(threshold=sys.maxsize)
        # print(fitted_loss[fitted_loss_sorted_indices])

        # train_x = train_x[loss_good_indices, :, :]
        # train_y = train_y[loss_good_indices]

        print('retraining data size:', len(loss_good_indices), len(fitted_loss))
        '''
        loss_weights = None
        if paradigm == 'MI':
            metrics = accuracy_score
        elif paradigm == 'ERP':
            metrics = balanced_accuracy_score

            # weighted CrossEntropy Loss
            loss_weights = []
            ar_unique, cnts_class = np.unique(y, return_counts=True)
            print("labels:", ar_unique)
            print("Counts:", cnts_class)
            loss_weights.append(1.0)
            loss_weights.append(cnts_class[0] / cnts_class[1])
            print(loss_weights)
            loss_weights = torch.Tensor(loss_weights)
            try:
                loss_weights = loss_weights.cuda()
            except:
                print('no cuda device for loss_weights')
        elif paradigm == 'Seizure':
            metrics = None

            # weighted CrossEntropy Loss
            loss_weights = []
            ar_unique, cnts_class = np.unique(train_y, return_counts=True)
            print("labels:", ar_unique)
            print("Counts:", cnts_class)
            loss_weights.append(1.0)
            loss_weights.append(cnts_class[0] / cnts_class[1])
            print(loss_weights)
            loss_weights = torch.Tensor(loss_weights)
            loss_weights = loss_weights.cuda()

            '''
            # Random Upsampling
            shapes = train_x.shape
            train_x = train_x.reshape(train_x.shape[0], -1)
            print(shapes, train_x.shape)
            train_x, train_y = apply_randup(train_x, train_y)
            train_x = train_x.reshape(-1, shapes[1], shapes[2])
            '''

            '''
            # Random Downsampling
            shapes = train_x.shape
            train_x = train_x.reshape(train_x.shape[0], -1)
            print(shapes, train_x.shape)
            train_x_all = []
            train_y_all = []
            num_trials_s = len(train_x) // (num_subjects - 1)
            for s in range(num_subjects - 1):
                train_x_down, train_y_down = apply_randdown(train_x[num_trials_s * s: num_trials_s * (s + 1)], train_y[num_trials_s * s: num_trials_s * (s + 1)])
                train_x_all.append(train_x_down)
                train_y_all.append(train_y_down)
            train_x = np.concatenate(train_x_all)
            train_y = np.concatenate(train_y_all).reshape(-1, )
            train_x = train_x.reshape(-1, shapes[1], shapes[2])
            print('train_x, train_y, test_x, test_y.shape', train_x.shape, train_y.shape, test_x.shape, test_y.shape)
            '''
        seed_arr = np.arange(3)
        rand_init_scores = []
        accs, sens, specs, aucs, f1s, bcas = [], [], [], [], [], []
        for seed in seed_arr:
            '''
            model_list = []
            for f in range(5):
                model = EEGNet_feature(n_classes=class_out,
                               Chans=train_x.shape[1],
                               Samples=train_x.shape[2],
                               kernLenght=int(sample_rate // 2),
                               F1=4,
                               D=2,
                               F2=8,
                               dropoutRate=0.25,
                               norm_rate=0.5)
                model_list.append(model)

            '''
            if approach == 'EEGNet':
                model = EEGNet(n_classes=class_out,
                               Chans=train_x.shape[1],
                               Samples=train_x.shape[2],
                               kernLenght=int(sample_rate // 2),
                               F1=4,
                               D=2,
                               F2=8,
                               dropoutRate=0.5, # TODO within
                               norm_rate=0.5)
            elif approach == 'EEGWaveNet':
                model = EEGWaveNet(n_chans=ch_num,
                                   n_classes=class_out)
            elif approach == 'BEEGNet':
                model = BEEGNet(n_classes=class_out,
                               Chans=train_x.shape[1],
                               Samples=train_x.shape[2],
                               kernLenght=int(sample_rate // 2),
                               F1=4,
                               D=2,
                               F2=8,
                               dropoutRate=0.25,
                               norm_rate=0.5)
            '''
            elif approach == 'CE_stSENet':
                model = CE_stSENet(inc=ch_num,
                                   class_num=class_out,
                                   si=sample_rate)
            '''

            '''
            # torch-summary
            print(summary(model, (1, 8, 206))) # (1, chan, ts)
            sys.exit(0)
            '''

            if paradigm == 'Seizure':
                acc, sen, spec, auc, f1, bca = nn_fixepoch(model=model,
                                          learning_rate=0.001,
                                          num_iterations=10,
                                          metrics=metrics,
                                          cuda=True,
                                          cuda_device_id=cuda_device_id,
                                          seed=int(seed),
                                          dataset=dataset,
                                          model_name=approach,
                                          test_subj_id=i,
                                          label_probs=False,
                                          valid_percentage=0,
                                          train_x=train_x,
                                          train_y=train_y,
                                          test_x=test_x,
                                          test_y=test_y,
                                          loss_weights=loss_weights)
                accs.append(acc)
                sens.append(sen)
                specs.append(spec)
                aucs.append(auc)
                f1s.append(f1)
                bcas.append(bca)
            else:
                rand_init_score = nn_fixepoch(model=model,
                                              learning_rate=0.001,
                                              num_iterations=100,
                                              metrics=metrics,
                                              cuda=True,
                                              cuda_device_id=cuda_device_id,
                                              seed=int(seed),
                                              dataset=dataset,
                                              model_name='EEGNet',
                                              test_subj_id=i,
                                              label_probs=False,
                                              valid_percentage=0,
                                              train_x=train_x,
                                              train_y=train_y,
                                              test_x=test_x,
                                              test_y=test_y,
                                              loss_weights=loss_weights)
                rand_init_scores.append(rand_init_score)
        print('subj rand_init_scores:', rand_init_scores)
        if paradigm == 'Seizure':
            print('subj all metrics:', accs, sens, specs, aucs, f1s, bcas)
            mind = 0
            for mets in [accs, sens, specs, aucs, f1s, bcas]:
                score = np.round(np.average(mets), 5)
                scores_arr_all[mind].append(mets)
                print(eval_metrics[mind], score)
                mind += 1
        else:
            score = np.round(np.average(rand_init_scores), 5)
            scores_arr.append(rand_init_scores)
            print('acc:', score)
    if paradigm == 'Seizure':
        mind = 0
        for scores_arr in scores_arr_all:
            scores_arr = np.stack(scores_arr)
            print('#' * 40)
            print('metrics:', eval_metrics[mind])
            mind += 1

            print('all scores', scores_arr)
            all_avgs = np.average(scores_arr, 1).round(3)
            print('all avgs', all_avgs)
            subj_stds = np.std(scores_arr, 1).round(3)
            print('sbj stds', subj_stds)
            all_avg = np.average(np.average(scores_arr, 0)).round(3)
            print('all avg', all_avg)
            all_std = np.std(np.average(scores_arr, 0)).round(3)
            print('all std', all_std)
    else:
        scores_arr = np.stack(scores_arr)
        print('#' * 40)
        scores_arr *= 100

        print('all scores', scores_arr)
        all_avgs = np.average(scores_arr, 1).round(3)
        print('all avgs', all_avgs)
        subj_stds = np.std(scores_arr, 1).round(3)
        print('sbj stds', subj_stds)
        all_avg = np.average(np.average(scores_arr, 0)).round(3)
        print('all avg', all_avg)
        all_std = np.std(np.average(scores_arr, 0)).round(3)
        print('all std', all_std)


def eeg_dnn_ms(dataset, info, align, approach, cuda_device_id):
    X, y, num_subjects, paradigm, sample_rate, ch_num = data_loader(dataset)
    print('sample rate:', sample_rate)

    if align:
        X = data_alignment(X, num_subjects)

    scores_arr = []
    for i in range(num_subjects):
        train_x, train_y, test_x, test_y = traintest_split_cross_subject(dataset, X, y, num_subjects, i)
        print('train_x, train_y, test_x, test_y.shape', train_x.shape, train_y.shape, test_x.shape, test_y.shape)

        class_out = len(np.unique(train_y))

        loss_weights = None
        if paradigm == 'MI':
            metrics = accuracy_score
        elif paradigm == 'ERP':
            metrics = balanced_accuracy_score

            # weighted CrossEntropy Loss
            loss_weights = []
            ar_unique, cnts_class = np.unique(y, return_counts=True)
            print("labels:", ar_unique)
            print("Counts:", cnts_class)
            loss_weights.append(1.0)
            loss_weights.append(cnts_class[0] / cnts_class[1])
            print(loss_weights)
            loss_weights = torch.Tensor(loss_weights)
            loss_weights = loss_weights.to(torch.device('cuda:' + str(cuda_device_id)))

        if dataset == 'BNCI2014001':
            deep_feature_size = 248
        elif dataset == 'BNCI2014002':
            deep_feature_size = 640
        elif dataset == 'BNCI2015001':
            deep_feature_size = 640
        elif dataset == 'BNCI2014008':
            deep_feature_size = 48
        elif dataset == 'BNCI2014009':
            deep_feature_size = 48
        elif dataset == 'BNCI2015003':
            deep_feature_size = 48

        seed_arr = np.arange(5)
        rand_init_scores = []
        for seed in seed_arr:
            model = EEGNet_feature(n_classes=class_out,
                                   Chans=train_x.shape[1],
                                   Samples=train_x.shape[2],
                                   kernLenght=int(sample_rate // 2),
                                   F1=4,
                                   D=2,
                                   F2=8,
                                   dropoutRate=0.25,
                                   norm_rate=0.5)

            rand_init_score = nn_fixepoch_ms(model=model,
                                             learning_rate=0.001,
                                             num_iterations=100,
                                             metrics=metrics,
                                             cuda=True,
                                             cuda_device_id=cuda_device_id,
                                             seed=int(seed),
                                             dataset=dataset,
                                             model_name='EEGNet',
                                             test_subj_id=i,
                                             label_probs=False,
                                             train_x=train_x,
                                             train_y=train_y,
                                             test_x=test_x,
                                             test_y=test_y,
                                             loss_weights=loss_weights,
                                             num_sources=num_subjects - 1,
                                             deep_feature_size=deep_feature_size)
            # loss_weights=loss_weights)

            rand_init_scores.append(rand_init_score)
        print('subj rand_init_scores:', rand_init_scores)
        score = np.round(np.average(rand_init_scores), 5)
        scores_arr.append(rand_init_scores)
        print('acc:', score)

    scores_arr = np.stack(scores_arr)
    print('#' * 40)
    scores_arr *= 100

    print('all scores', scores_arr)
    all_avgs = np.average(scores_arr, 1).round(3)
    print('all avgs', all_avgs)
    subj_stds = np.std(scores_arr, 1).round(3)
    print('sbj stds', subj_stds)
    all_avg = np.average(np.average(scores_arr, 0)).round(3)
    print('all avg', all_avg)
    all_std = np.std(np.average(scores_arr, 0)).round(3)
    print('all std', all_std)


def eeg_dnn_test(dataset, info, align, approach, cuda_device_id):
    X, y, num_subjects, paradigm, sample_rate, ch_num = data_loader(dataset)
    print('sample rate:', sample_rate)

    unaligned_X = X.copy()

    if align:
        X = data_alignment(X, num_subjects)

    scores_arr = []
    std_arr = []
    for i in range(num_subjects):
        train_x, train_y, test_x, test_y = traintest_split_cross_subject(dataset, X, y, num_subjects, i)
        print('train_x, train_y, test_x, test_y.shape', train_x.shape, train_y.shape, test_x.shape, test_y.shape)

        if paradigm == 'MI' and approach == 'transform':
            csp = CSP(n_components=ch_num)  # TODO
            csp.fit_transform(train_x, train_y)
            train_x = csp.X_filtered_fitted
            # assert csp.X_filtered_fitted == csp.X_filtered_transformed, 'ERROR CSP Transform!'
            csp.transform(test_x)
            test_x = csp.X_filtered_transformed
            print('Training/Test split after CSP:', train_x.shape, test_x.shape)
        if paradigm == 'ERP' and approach == 'transform':
            xdawn = Xdawn(n_components=X.shape[1])  # number of channels
            train_x_epochs = mne.EpochsArray(train_x, info)
            test_x_epochs = mne.EpochsArray(test_x, info)
            train_x = xdawn.fit_transform(train_x_epochs)  # unsupervised
            test_x = xdawn.transform(test_x_epochs)
            # train_x = train_x_xdawn.reshape(train_x_xdawn.shape[0], -1)
            # test_x = test_x_xdawn.reshape(test_x_xdawn.shape[0], -1)
            print('Training/Test split after xDAWN:', train_x.shape, test_x.shape)
        class_out = len(np.unique(train_y))
        loss_weights = None
        if paradigm == 'MI':
            metrics = accuracy_score
        elif paradigm == 'ERP':
            metrics = balanced_accuracy_score

            # weighted CrossEntropy Loss
            loss_weights = []
            ar_unique, cnts_class = np.unique(y, return_counts=True)
            print("labels:", ar_unique)
            print("Counts:", cnts_class)
            loss_weights.append(1.0)
            loss_weights.append(cnts_class[0] / cnts_class[1])
            print(loss_weights)
            loss_weights = torch.Tensor(loss_weights)
            loss_weights = loss_weights.to(torch.device('cuda:' + str(cuda_device_id)))
        seed_arr = np.arange(5)
        rand_init_scores = []
        for seed in seed_arr:
            model = EEGNet(n_classes=class_out,
                           Chans=train_x.shape[1],  # 10,  #
                           Samples=train_x.shape[2],
                           kernLenght=int(sample_rate // 2),
                           F1=4,
                           D=2,
                           F2=8,
                           dropoutRate=0.25,
                           norm_rate=0.5)
            model = torch.load('./runs/' + str(dataset) + '/' + 'EEGNet' + '_' + str(i) +
                               '_epoch_100_seed_' + str(seed) + '.ckpt')
            rand_init_score = nn_fixepoch(model=model,
                                          learning_rate=0.001,
                                          num_iterations=0,
                                          metrics=metrics,
                                          cuda=True,
                                          cuda_device_id=cuda_device_id,
                                          seed=int(seed),
                                          dataset=dataset,
                                          model_name='EEGNet',
                                          test_subj_id=i,
                                          label_probs=False,
                                          valid_percentage=0,
                                          train_x=train_x,
                                          train_y=train_y,
                                          test_x=test_x,
                                          test_y=test_y,
                                          loss_weights=loss_weights)
            # loss_weights=loss_weights)

            rand_init_scores.append(rand_init_score)
        print('subj rand_init_scores:', rand_init_scores)
        score = np.round(np.average(rand_init_scores), 5)
        scores_arr.append(rand_init_scores)
        print('acc:', score)

    scores_arr = np.stack(scores_arr)
    print('#' * 40)
    scores_arr *= 100

    print('all scores', scores_arr)
    all_avgs = np.average(scores_arr, 1).round(3)
    print('all avgs', all_avgs)
    subj_stds = np.std(scores_arr, 1).round(3)
    print('sbj stds', subj_stds)
    all_avg = np.average(np.average(scores_arr, 0)).round(3)
    print('all avg', all_avg)
    all_std = np.std(np.average(scores_arr, 0)).round(3)
    print('all std', all_std)


def load_deep_feature_from_model(dataset, pre_model_arch, cuda_device_id, test_subj, train_x, test_x, sample_rate):
    model_dir = './runs/' + dataset + '/'

    device = torch.device('cpu')
    if cuda_device_id:
        torch.cuda.manual_seed(seed)
        device = torch.device('cuda:' + str(cuda_device_id))
        print('using cuda...')

    tensor_train_x = torch.from_numpy(train_x).to(torch.float32)
    tensor_train_x = tensor_train_x.unsqueeze_(3).permute(0, 3, 1, 2)
    train_dataset = TensorDataset(tensor_train_x)
    train_loader = DataLoader(train_dataset, batch_size=256)

    tensor_test_x = torch.from_numpy(test_x).to(torch.float32)
    tensor_test_x = tensor_test_x.unsqueeze_(3).permute(0, 3, 1, 2)
    test_dataset = TensorDataset(tensor_test_x)
    test_loader = DataLoader(test_dataset, batch_size=256)

    model_path = model_dir + pre_model_arch + '_testsubjID_' + str(test_subj) + '_epoch_100.ckpt'
    model = torch.load(model_path)
    model.cuda()
    model.eval()
    model_dict = model.state_dict()

    if pre_model_arch == 'EEGNet':

        print('loading only feature extractor of trained EEGNet model...')
        model_feature = EEGNet_feature(n_classes=-1,
                                       Chans=train_x.shape[1],
                                       Samples=train_x.shape[2],
                                       kernLenght=int(sample_rate // 2),
                                       F1=4,
                                       D=2,
                                       F2=8,
                                       dropoutRate=0.25,
                                       norm_rate=0.5)
        model_feature.cuda()
        model_feature.eval()
        model_feature_dict = model_feature.state_dict()

        updated_dict = {}
        for k, v in model_dict.items():
            if not k.startswith('classifier_block'):
                # print('loading ', k, v.shape)
                updated_dict[k] = v
        model_feature_dict.update(updated_dict)

        deep_feature_train_x = []
        deep_feature_test_x = []

        for i, x in enumerate(train_loader):
            x = x[0].cuda()
            outputs = model_feature(x)
            outputs = outputs.cpu().detach().numpy()
            deep_feature_train_x.append(outputs)
        for i, x in enumerate(test_loader):
            x = x[0].cuda()
            outputs = model_feature(x)
            outputs = outputs.cpu().detach().numpy()
            deep_feature_test_x.append(outputs)

        deep_feature_train_x = np.concatenate(deep_feature_train_x, axis=0)
        deep_feature_test_x = np.concatenate(deep_feature_test_x, axis=0)

    return deep_feature_train_x, deep_feature_test_x


def load_and_predict_from_model(dataset, cuda_device_id, pre_model_arch, test_subj, test_x):
    model_dir = './runs/' + dataset + '/'

    device = torch.device('cpu')
    if cuda_device_id:
        torch.cuda.manual_seed(seed)
        device = torch.device('cuda:' + str(cuda_device_id))
        print('using cuda...')

    model_path = model_dir + pre_model_arch + '_testsubjID_' + str(test_subj) + '_epoch_100.ckpt'
    print(model_path)
    model = torch.load(model_path)
    model.cuda()
    model.eval()

    tensor_test_x = torch.from_numpy(test_x).to(torch.float32)
    tensor_test_x = tensor_test_x.unsqueeze_(3).permute(0, 3, 1, 2)
    test_dataset = TensorDataset(tensor_test_x)
    test_loader = DataLoader(test_dataset, batch_size=256)

    predicted_prob = []
    m = torch.nn.Softmax(dim=1)
    for i, x in enumerate(test_loader):
        x = x[0].cuda()
        outputs_prob = m(model(x))
        outputs_prob = outputs_prob.cpu().detach().numpy()

        predicted_prob.append(outputs_prob)
    predicted_prob = np.concatenate(predicted_prob, axis=0)
    return predicted_prob


def load_and_calculate_loss_from_model(dataset, cuda_device_id, pre_model_arch, test_subj, test_x, test_y):
    model_dir = './runs/' + dataset + '/'

    device = torch.device('cpu')
    if cuda_device_id:
        torch.cuda.manual_seed(seed)
        device = torch.device('cuda:' + str(cuda_device_id))
        print('using cuda...')

    model_path = model_dir + pre_model_arch + '_testsubjID_' + str(test_subj) + '_epoch_100.ckpt'
    model = torch.load(model_path)
    model.cuda()
    model.eval()

    tensor_test_x, tensor_test_y = torch.from_numpy(test_x).to(
        torch.float32), torch.from_numpy(test_y.reshape(-1, )).to(torch.long)
    if pre_model_arch == 'EEGNet':
        tensor_test_x = tensor_test_x.unsqueeze_(3).permute(0, 3, 1, 2)
    test_dataset = TensorDataset(tensor_test_x, tensor_test_y)
    test_loader = DataLoader(test_dataset)

    criterion = torch.nn.CrossEntropyLoss()

    fitted_loss = []
    for i, (x, y) in enumerate(test_loader):
        x = x.cuda()
        y = y.cuda()
        outputs = model(x)
        # print(outputs, y)
        loss = criterion(outputs, y)
        loss = loss.cpu().item()

        fitted_loss.append(loss)
    return fitted_loss


def autoencoder_model(cuda_device_id, train_x, test_x):
    num_iterations = 100
    learning_rate = 0.001

    device = torch.device('cpu')
    if cuda_device_id:
        torch.cuda.manual_seed(seed)
        device = torch.device('cuda:' + str(cuda_device_id))
        print('using cuda...')

    tensor_train_x = torch.from_numpy(train_x).to(torch.float32)
    train_dataset = TensorDataset(tensor_train_x)
    train_loader = DataLoader(train_dataset, batch_size=256)

    tensor_test_x = torch.from_numpy(test_x).to(torch.float32)
    test_dataset = TensorDataset(tensor_test_x)
    test_loader = DataLoader(test_dataset, batch_size=256)

    model = Autoencoder(encoder_neurons=[train_x.shape[1], 64, 16],
                        decoder_neurons=[16, 64, train_x.shape[1]])
    model.cuda()

    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()

    # Train the model
    for epoch in range(num_iterations):
        total_loss = 0
        cnt = 0
        for i, x in enumerate(train_loader):
            # Forward pass
            x = x[0].cuda()
            outputs = model(x)
            loss = criterion(outputs, x)
            total_loss += loss
            cnt += 1

            # Backward and optimize
            opt.zero_grad()
            loss.backward()
            opt.step()
        out_loss = total_loss / cnt

        if (epoch + 1) % 10 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_iterations, out_loss))

    model.eval()

    # use encoder only
    model_encoder = Autoencoder_encoder(encoder_neurons=[train_x.shape[1], 64, 16])
    model_encoder_dict = model_encoder.state_dict()
    model_encoder.eval()
    model_encoder.cuda()

    updated_dict = {}
    for k, v in model.state_dict().items():
        if k.startswith('encoder'):
            # print('loading ', k, v.shape)
            updated_dict[k] = v
    model_encoder_dict.update(updated_dict)

    # Transform the data
    train_x_transformed = []
    test_x_transformed = []

    for i, x in enumerate(train_loader):
        x = x[0].cuda()
        outputs = model_encoder(x)
        outputs = outputs.cpu().detach().numpy()
        train_x_transformed.append(outputs)
    for i, x in enumerate(test_loader):
        x = x[0].cuda()
        outputs = model_encoder(x)
        outputs = outputs.cpu().detach().numpy()
        test_x_transformed.append(outputs)

    train_x_transformed = np.concatenate(train_x_transformed, axis=0)
    test_x_transformed = np.concatenate(test_x_transformed, axis=0)
    print('Autoencoder transformed:', train_x_transformed.shape, test_x_transformed.shape)

    return train_x_transformed, test_x_transformed


def rbm_model(train_x, test_x):
    tensor_train_x = torch.from_numpy(train_x).to(torch.float32)
    train_dataset = TensorDataset(tensor_train_x)
    train_loader = DataLoader(train_dataset, batch_size=256)

    tensor_test_x = torch.from_numpy(test_x).to(torch.float32)
    test_dataset = TensorDataset(tensor_test_x)
    test_loader = DataLoader(test_dataset, batch_size=256)

    nv = train_x.shape[0]
    nh = 32
    batch_size = 100
    rbm = RBM(nv, nh)

    nb_epoch = 10
    for epoch in range(1, nb_epoch + 1):
        train_loss = 0
        s = 0.
        for i, x in enumerate(train_loader):
            x = x[0]
            vk = x
            v0 = x
            ph0, _ = rbm.sample_h(v0)
            for k in range(10):
                _, hk = rbm.sample_h(vk)
                _, vk = rbm.sample_v(hk)
                vk[v0 < 0] = v0[v0 < 0]
            phk, _ = rbm.sample_h(vk)
            rbm.train(v0, vk, ph0, phk)
            train_loss += torch.mean(torch.abs(v0[v0 >= 0] - vk[v0 >= 0]))
            s += 1.
        print('epoch: ' + str(epoch) + ' loss: ' + str(train_loss / s))

    train_x_transformed = train_x
    test_x_transformed = test_x

    return train_x_transformed, test_x_transformed


def eeg_fusion(dataset, info, fusion, align, approach, pre_model_arch, cuda_device_id):
    X, y, num_subjects, paradigm, sample_rate, ch_num = data_loader(dataset)
    print('sample rate:', sample_rate)

    if paradigm == 'ERP':
        X_downsample = mne.filter.resample(X, down=4)

    if align:
        if paradigm == 'MI':
            '''
            if dataset == 'BNCI2014002':
                session_split = 0.625
            else:
                session_split = 0.5
            X = data_alignment_multiple_sessions(X, num_subjects, session_split=session_split)
            '''
            X = data_alignment(X, num_subjects)
        else:
            X = data_alignment(X, num_subjects)
            X_downsample = data_alignment(X_downsample, num_subjects)

    scores_arr = []

    for i in range(num_subjects):
        print('#' * 40)
        train_x, train_y, test_x, test_y = traintest_split_cross_subject(dataset, X, y, num_subjects, i)
        if paradigm == 'ERP':
            train_x_downsample, _, test_x_downsample, _ = traintest_split_cross_subject(dataset, X_downsample, y,
                                                                                        num_subjects, i)
        print('train_x, train_y, test_x, test_y.shape', train_x.shape, train_y.shape, test_x.shape, test_y.shape)

        if fusion == 'feature' or fusion == 'Autoencoder' or fusion == 'RBM':
            deep_feature_train_x, deep_feature_test_x = load_deep_feature_from_model(dataset, pre_model_arch,
                                                                                     cuda_device_id, i,
                                                                                     train_x, test_x, sample_rate)

            # PCA for deep features
            # deep_feature_train_x, deep_feature_test_x = apply_pca(deep_feature_train_x, deep_feature_test_x, 0.95)

        if paradigm == 'MI':
            # CSP
            csp = CSP(n_components=10)  # TODO modify this
            csp.fit_transform(train_x, train_y)
            train_x_csp = csp.X_filtered_fitted
            # assert csp.X_filtered_fitted == csp.X_filtered_transformed, 'ERROR CSP Transform!'
            csp.transform(test_x)
            test_x_csp = csp.X_filtered_transformed
            print('Training/Test split after CSP:', train_x_csp.shape, test_x_csp.shape)

            '''
            csp = CSP(n_components=10)
            train_x_csp = csp.fit_transform(train_x, train_y)
            test_x_csp = csp.transform(test_x)
            print('Training/Test split after CSP:', train_x_csp.shape, test_x_csp.shape)
            '''

            """
            X_fea, _, _, _, _, _ = data_loader_feature(dataset)
            train_x_fea, _, test_x_fea, _ = traintest_split_cross_subject(dataset, X_fea, y, num_subjects, i)

            ranking = [27, 32, 31, 30, 28, 29, 36, 26, 39, 40, 47, 25, 42, 34, 37, 35, 33, 67, 41, 60, 63, 43, 62, 64
                , 55, 38, 46, 69, 65, 68, 66, 56, 73, 17, 16, 15, 24, 2, 14, 54, 61, 10, 51, 13, 23, 11, 3, 72
                , 57, 7, 6, 49, 9, 1, 20, 8, 53, 18, 59, 22, 71, 12, 48, 21, 0, 4, 19, 52, 45, 58, 70, 50
                , 5, 44]

            feature_num = 50

            a = []
            inds_feature = ranking[(-1 * feature_num):]
            # print(inds_feature)
            for k in range(feature_num):
                inds = np.arange(ch_num, dtype=int) + inds_feature[k] * ch_num
                a.append(inds)
            a = np.concatenate(a)

            # print(a)
            # input('')

            train_x_fea = train_x_fea[:, a]
            test_x_fea = test_x_fea[:, a]

            #train_x_fea = np.mean(train_x_fea.reshape(train_x_fea.shape[0], feature_num, -1), axis=2)
            #test_x_fea = np.mean(test_x_fea.reshape(test_x_fea.shape[0], feature_num, -1), axis=2)

            train_x_fea = train_x_fea.reshape(train_x_fea.shape[0], -1)
            test_x_fea = test_x_fea.reshape(test_x_fea.shape[0], -1)

            #print(test_x_fea)

            # z-score standardization
            print('applying z-score train_x:', train_x_fea.shape, ' test_x:', test_x_fea.shape)
            train_x_fea, test_x_fea = apply_zscore(train_x_fea, test_x_fea, num_subjects)

            #print(test_x_fea)
            #sys.exit(0)

            #train_x_fea = train_x_fea.reshape(train_x_fea.shape[0], ch_num, -1)
            #test_x_fea = test_x_fea.reshape(test_x_fea.shape[0], ch_num, -1)

            train_x_fea = train_x_fea.reshape(train_x_fea.shape[0], feature_num, -1)
            test_x_fea = test_x_fea.reshape(test_x_fea.shape[0], feature_num, -1)

            """

            print('train_x_fea, test_x_fea shape:', train_x_csp.shape, test_x_csp.shape)

            if fusion == 'decision':
                # decision fusion
                pred_probs_classic = ml_classifier(approach, True, train_x_csp, train_y, test_x_csp)
                pred_probs_dnn = load_and_predict_from_model(dataset, cuda_device_id, pre_model_arch, i, test_x)
                pred_probs_ensemble = pred_probs_classic + pred_probs_dnn
                pred = np.argmax(pred_probs_ensemble, axis=1)

                pred_classic = ml_classifier(approach, False, train_x_csp, train_y, test_x_csp)
                print('CSP only:', np.round(accuracy_score(test_y, pred_classic), 5))
                score = np.round(accuracy_score(test_y, pred), 5)
            elif fusion == 'feature' or fusion == 'Autoencoder' or fusion == 'RBM':
                # feature fusion
                print('before feature fusion:', train_x_csp.shape, deep_feature_train_x.shape, test_x_csp.shape,
                      deep_feature_test_x.shape)
                fusion_train_x = np.concatenate([train_x_csp, deep_feature_train_x], axis=1)
                fusion_test_x = np.concatenate([test_x_csp, deep_feature_test_x], axis=1)
                print('after feature fusion:', fusion_train_x.shape, fusion_test_x.shape)

                if fusion == 'Autoencoder':
                    fusion_train_x, fusion_test_x = autoencoder_model(cuda_device_id, fusion_train_x, fusion_test_x)
                elif fusion == 'RBM':
                    fusion_train_x, fusion_test_x = rbm_model(fusion_train_x, fusion_test_x)

                # classifier
                if approach != 'FC':
                    # classifier
                    pred = ml_classifier(approach, False, fusion_train_x, train_y, fusion_test_x)
                    score = np.round(accuracy_score(test_y, pred), 5)
                else:
                    # FC model
                    feature_in = fusion_train_x.shape[1]
                    class_out = len(np.unique(y))

                    score = nn_fixepoch(model=FC(nn_in=feature_in, nn_out=class_out),
                                        learning_rate=0.001,
                                        num_iterations=100,
                                        metrics=accuracy_score,
                                        cuda=True,
                                        cuda_device_id=cuda_device_id,
                                        seed=42,
                                        dataset=dataset,
                                        model_name='FC',
                                        test_subj_id=i,
                                        label_probs=False,
                                        valid_percentage=0,
                                        train_x=fusion_train_x,
                                        train_y=train_y,
                                        test_x=fusion_test_x,
                                        test_y=test_y)
                    '''
                    if dataset == 'BNCI2014001':
                        deep_feature_size = 248
                    elif dataset == 'BNCI2014002':
                        deep_feature_size = 640
                    elif dataset == 'MI1':
                        deep_feature_size = 72
                    score = nn_fixepoch_doubleinput(model=FC_middlecat(nn_out=class_out, deep_feature_size=deep_feature_size, trad_feature_size=10),
                        learning_rate=0.001,
                        num_iterations=100,
                        metrics=accuracy_score,
                        cuda=True,
                        cuda_device_id=cuda_device_id,
                        seed=42,
                        dataset=dataset,
                        model_name='FC',
                        test_subj_id=i,
                        label_probs=False,
                        valid_percentage=0,
                        train_x1=deep_feature_train_x,
                        train_x2=train_x_csp,
                        train_y=train_y,
                        test_x1=deep_feature_test_x,
                        test_x2=test_x_csp,
                        test_y=test_y)
                    '''
            elif fusion == 'middle':
                if dataset == 'BNCI2014001':
                    feature_deep_dim = 248 * 2  # + ch_num * 4  # 248
                elif dataset == 'BNCI2014002':
                    feature_deep_dim = 640 * 2  # + ch_num * 4 # 640
                elif dataset == 'MI1':
                    feature_deep_dim = 72
                elif dataset == 'BNCI2015001':
                    feature_deep_dim = 640 * 2  # + ch_num * 4  # 640
                class_out = len(np.unique(train_y))
                '''
                # using prediction on training set of traditional approach to guide NN learning
                # suppress cross-entropy loss weights
                train_y_bool = np.load('./files/' + dataset + '_csp_pred_classification_testsubj_' + str(i) + '_.npy')
                train_y_prob = np.load('./files/' + dataset + '_csp_pred_testsubj_' + str(i) + '_.npy')
                out_y = np.zeros((len(train_y), class_out))
                assert len(train_y_bool) == len(train_y), 'Wrong file train_y_bool shape!'
                for p in range(len(train_y)):

                    # out_y[p, 0] = 0.1
                    # out_y[p, 1] = 0.1
                    if train_y[p] == 0:
                        out_y[p, 0] = (1 + train_y_prob[p, 0]) / 2
                        out_y[p, 1] = 1 - out_y[p, 0]
                    else:
                        out_y[p, 1] = (1 + train_y_prob[p, 1]) / 2
                        out_y[p, 0] = 1 - out_y[p, 1]

                train_y = out_y
                '''
                metrics = accuracy_score
                seed_arr = np.arange(5)
                rand_init_scores = []
                std_arr = []
                for seed in seed_arr:
                    model_data = EEGNet_feature(n_classes=class_out,
                                                Chans=train_x.shape[1],
                                                Samples=train_x.shape[2],
                                                kernLenght=int(sample_rate // 2),
                                                F1=4,
                                                D=2,
                                                F2=8,
                                                dropoutRate=0.5,
                                                norm_rate=0.5)
                    model_knowledge = EEGNet_feature(n_classes=class_out,
                                                     Chans=10,  # TODO
                                                     Samples=train_x.shape[2],
                                                     kernLenght=int(sample_rate // 2),
                                                     F1=4,
                                                     D=2,
                                                     F2=8,
                                                     dropoutRate=0.5,
                                                     norm_rate=0.5)
                    rand_init_score = nn_fixepoch_siamesefusion(model_data=model_data,
                                                                model_knowledge=model_knowledge,
                                                                learning_rate=0.001,
                                                                num_iterations=100,
                                                                metrics=metrics,
                                                                cuda=True,
                                                                cuda_device_id=cuda_device_id,
                                                                seed=int(seed),
                                                                dataset=dataset,
                                                                model_name='EEGNet',
                                                                test_subj_id=i,
                                                                label_probs=False,
                                                                valid_percentage=0,
                                                                train_x=train_x,
                                                                train_y=train_y,
                                                                test_x=test_x,
                                                                test_y=test_y,
                                                                middle_feature_train_x=train_x_csp,
                                                                middle_feature_test_x=test_x_csp,
                                                                feature_deep_dim=feature_deep_dim,
                                                                class_out=class_out,
                                                                ch_num=ch_num)
                    rand_init_scores.append(rand_init_score)
                """
                for seed in seed_arr:
                    model_data = EEGNet_feature(n_classes=class_out,
                                           Chans=train_x.shape[1],
                                           Samples=train_x.shape[2],
                                           kernLenght=int(sample_rate // 2),
                                           F1=4,  # 4 to 2
                                           D=2,
                                           F2=8,  # 8 to 4
                                           dropoutRate=0.25,
                                           norm_rate=0.5)
                    model_knowledge = ConvChannelWise(nn_deep=ch_num,
                                            nn_out=class_out,
                                            in_channels=feature_num,
                                            out_channels=4,
                                            bias=False,
                                            layer=1)
                    rand_init_score = nn_fixepoch_middlecat(model_data=model_data,
                                                            model_knowledge=model_knowledge,
                                                            learning_rate=0.001,
                                                            num_iterations=100,
                                                            metrics=metrics,
                                                            cuda=True,
                                                            cuda_device_id=cuda_device_id,
                                                            seed=int(seed),
                                                            dataset=dataset,
                                                            model_name='EEGNet',
                                                            test_subj_id=i,
                                                            label_probs=False,
                                                            valid_percentage=0,
                                                            train_x=train_x,
                                                            train_y=train_y,
                                                            test_x=test_x,
                                                            test_y=test_y,
                                                            middle_feature_train_x=train_x_fea,
                                                            middle_feature_test_x=test_x_fea,
                                                            feature_deep_dim=feature_deep_dim,
                                                            class_out=class_out,
                                                            ch_num=ch_num)

                    rand_init_scores.append(rand_init_score)
                """
                print('subj rand_init_scores:', rand_init_scores)
                score = np.round(np.average(rand_init_scores), 5)
                std = np.round(np.std(rand_init_scores), 5)
                std_arr.append(std)

            print('acc:', score)
        elif paradigm == 'ERP':
            # xDAWN
            # xdawn = Xdawn(n_components=X_downsample.shape[1])  # number of channels
            # train_x_epochs = mne.EpochsArray(train_x_downsample, info)
            # test_x_epochs = mne.EpochsArray(test_x_downsample, info)
            xdawn = Xdawn(n_components=X.shape[1])  # number of channels
            train_x_epochs = mne.EpochsArray(train_x, info)
            test_x_epochs = mne.EpochsArray(test_x, info)
            train_x_xdawn = xdawn.fit_transform(train_x_epochs)  # unsupervised
            test_x_xdawn = xdawn.transform(test_x_epochs)
            # train_x_xdawn = train_x_xdawn.reshape(train_x_xdawn.shape[0], -1)
            # test_x_xdawn = test_x_xdawn.reshape(test_x_xdawn.shape[0], -1)
            print('Training/Test split after xDAWN:', train_x_xdawn.shape, test_x_xdawn.shape)

            if fusion == 'middle':
                if dataset == 'BNCI2014008':
                    feature_deep_dim = 48 * 2
                elif dataset == 'BNCI2014009':
                    feature_deep_dim = 48 * 2
                elif dataset == 'BNCI2015003':
                    feature_deep_dim = 48 * 2
                class_out = len(np.unique(train_y))

                loss_weights = []
                ar_unique, cnts_class = np.unique(y, return_counts=True)
                print("labels:", ar_unique)
                print("Counts:", cnts_class)
                loss_weights.append(1.0)
                loss_weights.append(cnts_class[0] / cnts_class[1])
                print(loss_weights)
                loss_weights = torch.Tensor(loss_weights)
                loss_weights = loss_weights.to(torch.device('cuda:' + str(cuda_device_id)))

                metrics = balanced_accuracy_score
                seed_arr = np.arange(5)
                rand_init_scores = []
                std_arr = []
                for seed in seed_arr:
                    model_data = EEGNet_feature(n_classes=class_out,
                                                Chans=train_x.shape[1],
                                                Samples=train_x.shape[2],
                                                kernLenght=int(sample_rate // 2),
                                                F1=4,
                                                D=2,
                                                F2=8,
                                                dropoutRate=0.5,
                                                norm_rate=0.5)
                    model_knowledge = EEGNet_feature(n_classes=class_out,
                                                     Chans=train_x.shape[1],
                                                     Samples=train_x.shape[2],
                                                     kernLenght=int(sample_rate // 2),
                                                     F1=4,
                                                     D=2,
                                                     F2=8,
                                                     dropoutRate=0.5,
                                                     norm_rate=0.5)
                    rand_init_score = nn_fixepoch_siamesefusion(model_data=model_data,
                                                                model_knowledge=model_knowledge,
                                                                learning_rate=0.001,
                                                                num_iterations=100,
                                                                metrics=metrics,
                                                                cuda=True,
                                                                cuda_device_id=cuda_device_id,
                                                                seed=int(seed),
                                                                dataset=dataset,
                                                                model_name='EEGNet',
                                                                test_subj_id=i,
                                                                label_probs=False,
                                                                valid_percentage=0,
                                                                train_x=train_x,
                                                                train_y=train_y,
                                                                test_x=test_x,
                                                                test_y=test_y,
                                                                middle_feature_train_x=train_x_xdawn,
                                                                middle_feature_test_x=test_x_xdawn,
                                                                feature_deep_dim=feature_deep_dim,
                                                                class_out=class_out,
                                                                ch_num=ch_num,
                                                                loss_weights=loss_weights)

                    rand_init_scores.append(rand_init_score)
                print('subj rand_init_scores:', rand_init_scores)
                score = np.round(np.average(rand_init_scores), 5)
                std = np.round(np.std(rand_init_scores), 5)
                std_arr.append(std)
            print('bca:', score)
        scores_arr.append(score)

    print('#' * 40)
    for i in range(len(scores_arr)):
        scores_arr[i] *= 100
    print('sbj scores', scores_arr)
    print('avg', np.round(np.average(scores_arr), 5))

    for i in range(len(std_arr)):
        std_arr[i] *= 100
    print('sbj stds', std_arr)
    print('std_randinit', np.round(np.average(std_arr), 5))


def eeg_SFN(dataset, info, fusion, align, approach, pre_model_arch, cuda_device_id):
    X, y, num_subjects, paradigm, sample_rate, ch_num = data_loader(dataset)
    print('sample rate:', sample_rate)

    # if paradigm == 'ERP':
    #    X_downsample = mne.filter.resample(X, down=4)

    if align:
        if paradigm == 'MI':
            X = data_alignment(X, num_subjects)
        else:
            X = data_alignment(X, num_subjects)
            # X_downsample = data_alignment(X_downsample, num_subjects)

    scores_arr = []
    std_arr = []

    for i in range(num_subjects):

        print('#' * 40)
        train_x, train_y, test_x, test_y = traintest_split_cross_subject(dataset, X, y, num_subjects, i)
        # if paradigm == 'ERP':
        #    train_x_downsample, _, test_x_downsample, _ = traintest_split_cross_subject(dataset, X_downsample, y,
        #                                                                                num_subjects, i)
        print('train_x, train_y, test_x, test_y.shape', train_x.shape, train_y.shape, test_x.shape, test_y.shape)

        if paradigm == 'MI':
            """
            # CSP
            csp = CSP(n_components=10) # 10  # TODO modify this
            csp.fit_transform(train_x, train_y)
            train_x_csp = csp.X_filtered_fitted
            csp.transform(test_x)
            test_x_csp = csp.X_filtered_transformed
            print('Training/Test split after CSP:', train_x_csp.shape, test_x_csp.shape)
            """

            X_fea, _, _, _, _, _ = data_loader_feature(dataset)
            train_x_fea, _, test_x_fea, _ = traintest_split_cross_subject(dataset, X_fea, y, num_subjects, i)

            ranking = [27, 32, 31, 30, 28, 29, 36, 26, 39, 40, 47, 25, 42, 34, 37, 35, 33, 67, 41, 60, 63, 43, 62, 64
                , 55, 38, 46, 69, 65, 68, 66, 56, 73, 17, 16, 15, 24, 2, 14, 54, 61, 10, 51, 13, 23, 11, 3, 72
                , 57, 7, 6, 49, 9, 1, 20, 8, 53, 18, 59, 22, 71, 12, 48, 21, 0, 4, 19, 52, 45, 58, 70, 50
                , 5, 44]

            feature_num = 74

            a = []
            inds_feature = ranking[(-1 * feature_num):]
            # print(inds_feature)
            for k in range(feature_num):
                inds = np.arange(ch_num, dtype=int) + inds_feature[k] * ch_num
                a.append(inds)
            a = np.concatenate(a)

            # print(a)
            # input('')

            train_x_fea = train_x_fea[:, a]
            test_x_fea = test_x_fea[:, a]

            # train_x_fea = np.mean(train_x_fea.reshape(train_x_fea.shape[0], feature_num, -1), axis=2)
            # test_x_fea = np.mean(test_x_fea.reshape(test_x_fea.shape[0], feature_num, -1), axis=2)

            train_x_fea = train_x_fea.reshape(train_x_fea.shape[0], -1)
            test_x_fea = test_x_fea.reshape(test_x_fea.shape[0], -1)

            # print(test_x_fea)

            # z-score standardization
            print('applying z-score train_x:', train_x_fea.shape, ' test_x:', test_x_fea.shape)
            train_x_fea, test_x_fea = apply_zscore(train_x_fea, test_x_fea, num_subjects)

            # print(test_x_fea)
            # sys.exit(0)

            # train_x_fea = train_x_fea.reshape(train_x_fea.shape[0], ch_num, -1)
            # test_x_fea = test_x_fea.reshape(test_x_fea.shape[0], ch_num, -1)

            train_x_csp = train_x_fea.reshape(train_x_fea.shape[0], ch_num, -1)
            test_x_csp = test_x_fea.reshape(test_x_fea.shape[0], ch_num, -1)

            print('train_x_fea, test_x_fea shape:', train_x_csp.shape, test_x_csp.shape)

            class_out = len(np.unique(train_y))

            metrics = accuracy_score
            seed_arr = np.arange(5)
            rand_init_scores = []
            for seed in seed_arr:
                """
                model = EEGNetSiameseFusion(n_classes=class_out,
                                            Chans=train_x.shape[1],
                                            Samples=train_x.shape[2],
                                            kernLenght=int(sample_rate // 2),
                                            F1=4,
                                            D=2,
                                            F2=8,
                                            dropoutRate=0.25,
                                            norm_rate=0.5,
                                            ch_num=10,
                                            return_representation=True)
                """
                if dataset == 'BNCI2014001':
                    feature_deep_dim = 248
                elif dataset == 'BNCI2014002':
                    feature_deep_dim = 640
                elif dataset == 'MI1':
                    feature_deep_dim = 72
                elif dataset == 'BNCI2015001':
                    feature_deep_dim = 640
                model = EEGNetCNNFusion(n_classes=class_out,
                                        Chans=train_x.shape[1],
                                        Samples=train_x.shape[2],
                                        kernLenght=int(sample_rate // 2),
                                        F1=4,
                                        D=2,
                                        F2=8,
                                        dropoutRate=0.25,
                                        norm_rate=0.5,
                                        deep_dim=feature_deep_dim)

                rand_init_score = nn_fixepoch_SFN(model=model,
                                                  learning_rate=0.001,
                                                  num_iterations=100,
                                                  metrics=metrics,
                                                  cuda=True,
                                                  cuda_device_id=cuda_device_id,
                                                  seed=int(seed),
                                                  dataset=dataset,
                                                  model_name='EEGNet',
                                                  test_subj_id=i,
                                                  label_probs=False,
                                                  valid_percentage=0,
                                                  train_x=train_x,
                                                  train_y=train_y,
                                                  test_x=test_x,
                                                  test_y=test_y,
                                                  middle_feature_train_x=train_x_csp,
                                                  middle_feature_test_x=test_x_csp,
                                                  class_out=class_out,
                                                  ch_num=ch_num)
                rand_init_scores.append(rand_init_score)

            print('subj rand_init_scores:', rand_init_scores)
            score = np.round(np.average(rand_init_scores), 5)
            print('acc:', score)
        elif paradigm == 'ERP':
            # xDAWN
            # xdawn = Xdawn(n_components=X_downsample.shape[1])  # number of channels
            # train_x_epochs = mne.EpochsArray(train_x_downsample, info)
            # test_x_epochs = mne.EpochsArray(test_x_downsample, info)
            xdawn = Xdawn(n_components=X.shape[1])  # number of channels
            train_x_epochs = mne.EpochsArray(train_x, info)
            test_x_epochs = mne.EpochsArray(test_x, info)
            train_x_xdawn = xdawn.fit_transform(train_x_epochs)  # unsupervised
            test_x_xdawn = xdawn.transform(test_x_epochs)
            # train_x_xdawn = train_x_xdawn.reshape(train_x_xdawn.shape[0], -1)
            # test_x_xdawn = test_x_xdawn.reshape(test_x_xdawn.shape[0], -1)
            print('Training/Test split after xDAWN:', train_x_xdawn.shape, test_x_xdawn.shape)

            class_out = len(np.unique(train_y))

            loss_weights = []
            ar_unique, cnts_class = np.unique(y, return_counts=True)
            print("labels:", ar_unique)
            print("Counts:", cnts_class)
            loss_weights.append(1.0)
            loss_weights.append(cnts_class[0] / cnts_class[1])
            print(loss_weights)
            loss_weights = torch.Tensor(loss_weights)
            loss_weights = loss_weights.to(torch.device('cuda:' + str(cuda_device_id)))

            metrics = balanced_accuracy_score
            seed_arr = np.arange(5)
            rand_init_scores = []

            for seed in seed_arr:
                model = EEGNetSiameseFusion(n_classes=class_out,
                                            Chans=train_x.shape[1],
                                            Samples=train_x.shape[2],
                                            kernLenght=int(sample_rate // 2),
                                            F1=4,
                                            D=2,
                                            F2=8,
                                            dropoutRate=0.25,
                                            norm_rate=0.5,
                                            return_representation=True)
                rand_init_score = nn_fixepoch_SFN(model=model,
                                                  learning_rate=0.001,
                                                  num_iterations=100,
                                                  metrics=metrics,
                                                  cuda=True,
                                                  cuda_device_id=cuda_device_id,
                                                  seed=int(seed),
                                                  dataset=dataset,
                                                  model_name='EEGNet',
                                                  test_subj_id=i,
                                                  label_probs=False,
                                                  valid_percentage=0,
                                                  train_x=train_x,
                                                  train_y=train_y,
                                                  test_x=test_x,
                                                  test_y=test_y,
                                                  middle_feature_train_x=train_x_xdawn,
                                                  middle_feature_test_x=test_x_xdawn,
                                                  class_out=class_out,
                                                  ch_num=ch_num,
                                                  loss_weights=loss_weights)
                rand_init_scores.append(rand_init_score)
            print('subj rand_init_scores:', rand_init_scores)
            score = np.round(np.average(rand_init_scores), 5)
            print('bca:', score)
        scores_arr.append(rand_init_scores)

    scores_arr = np.stack(scores_arr)
    print('#' * 40)
    scores_arr *= 100

    print('all scores', scores_arr)
    all_avgs = np.average(scores_arr, 1).round(3)
    print('all avgs', all_avgs)
    subj_stds = np.std(scores_arr, 1).round(3)
    print('sbj stds', subj_stds)
    all_avg = np.average(np.average(scores_arr, 0)).round(3)
    print('all avg', all_avg)
    all_std = np.std(np.average(scores_arr, 0)).round(3)
    print('all std', all_std)


def eeg_cotrain(dataset, info, align, approach, cuda_device_id):
    X, y, num_subjects, paradigm, sample_rate, ch_num = data_loader(dataset)
    print('sample rate:', sample_rate)

    if align:
        X = data_alignment(X, num_subjects)

    scores_arr = []
    for i in range(num_subjects):
        train_x, train_y, test_x, test_y = traintest_split_cross_subject(dataset, X, y, num_subjects, i)
        print('train_x, train_y, test_x, test_y.shape', train_x.shape, train_y.shape, test_x.shape, test_y.shape)

        if paradigm == 'MI':
            csp = CSP(n_components=10)
            csp.fit_transform(train_x, train_y)
            train_x_fea = csp.X_filtered_fitted
            csp.transform(test_x)
            test_x_fea = csp.X_filtered_transformed
            print('Training/Test split after CSP:', train_x_fea.shape, test_x_fea.shape)
        if paradigm == 'ERP':
            xdawn = Xdawn(n_components=X.shape[1])  # number of channels
            train_x_epochs = mne.EpochsArray(train_x, info)
            test_x_epochs = mne.EpochsArray(test_x, info)
            train_x_fea = xdawn.fit_transform(train_x_epochs)  # unsupervised
            test_x_fea = xdawn.transform(test_x_epochs)

            print('Training/Test split after xDAWN:', train_x_fea.shape, test_x_fea.shape)

        class_out = len(np.unique(train_y))

        loss_weights = None
        if paradigm == 'MI':
            metrics = accuracy_score
        elif paradigm == 'ERP':
            metrics = balanced_accuracy_score

            # weighted CrossEntropy Loss
            loss_weights = []
            ar_unique, cnts_class = np.unique(y, return_counts=True)
            print("labels:", ar_unique)
            print("Counts:", cnts_class)
            loss_weights.append(1.0)
            loss_weights.append(cnts_class[0] / cnts_class[1])
            print(loss_weights)
            loss_weights = torch.Tensor(loss_weights)
            loss_weights = loss_weights.to(torch.device('cuda:' + str(cuda_device_id)))

        seed_arr = np.arange(5)
        rand_init_scores = []
        for seed in seed_arr:

            if paradigm == 'MI':
                model_k = EEGNetxy(n_classes=class_out,
                                   Chans=10,
                                   Samples=train_x.shape[2],
                                   kernLenght=int(sample_rate // 2),
                                   F1=4,
                                   D=2,
                                   F2=8,
                                   dropoutRate=0.25,
                                   norm_rate=0.5)
            elif paradigm == 'ERP':
                model_k = EEGNetxy(n_classes=class_out,
                                   Chans=train_x.shape[1],
                                   Samples=train_x.shape[2],
                                   kernLenght=int(sample_rate // 2),
                                   F1=4,
                                   D=2,
                                   F2=8,
                                   dropoutRate=0.25,
                                   norm_rate=0.5)
            model_d = EEGNetxy(n_classes=class_out,
                               Chans=train_x.shape[1],
                               Samples=train_x.shape[2],
                               kernLenght=int(sample_rate // 2),
                               F1=4,
                               D=2,
                               F2=8,
                               dropoutRate=0.25,
                               norm_rate=0.5)

            rand_init_score = nn_cotrain(model_k=model_k,
                                         model_d=model_d,
                                         learning_rate=0.001,
                                         num_iterations=100,
                                         metrics=metrics,
                                         cuda=True,
                                         cuda_device_id=cuda_device_id,
                                         seed=int(seed),
                                         dataset=dataset,
                                         model_name='EEGNet',
                                         test_subj_id=i,
                                         label_probs=False,
                                         valid_percentage=0,
                                         train_x=train_x,
                                         train_y=train_y,
                                         test_x=test_x,
                                         test_y=test_y,
                                         feature_train_x=train_x_fea,
                                         feature_test_x=test_x_fea,
                                         class_out=class_out,
                                         ch_num=None,
                                         loss_weights=None)

            rand_init_scores.append(rand_init_score)
            print(np.round(rand_init_score, 5))
        print('subj rand_init_scores:', rand_init_scores)
        score = np.round(np.average(rand_init_scores), 5)
        scores_arr.append(rand_init_scores)
        print('acc:', score)

    scores_arr = np.stack(scores_arr)
    print('#' * 40)
    scores_arr *= 100

    print('all scores', scores_arr)
    all_avgs = np.average(scores_arr, 1).round(3)
    print('all avgs', all_avgs)
    subj_stds = np.std(scores_arr, 1).round(3)
    print('sbj stds', subj_stds)
    all_avg = np.average(np.average(scores_arr, 0)).round(3)
    print('all avg', all_avg)
    all_std = np.std(np.average(scores_arr, 0)).round(3)
    print('all std', all_std)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        cuda_device_id = str(sys.argv[1])
    else:
        cuda_device_id = -1
    try:
        device = torch.device('cuda:' + cuda_device_id)
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device_id
        print('using GPU')
    except:
        device = torch.device('cpu')
        print('using CPU')

    scores = []

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    #dataset_arr = ['NICU']
    #dataset_arr = ['CHSZ']
    #dataset_arr = ['PhysionetMI']
    #dataset_arr = ['Cho2017']
    dataset_arr = ['BNCI2014001', 'BNCI2014002', 'BNCI2015001']
    #dataset_arr = ['BNCI2014008', 'BNCI2014009', 'BNCI2015003']
    #dataset_arr = ['BNCI2014001', 'BNCI2014002', 'BNCI2015001', 'BNCI2014008', 'BNCI2014009', 'BNCI2015003']
    # dataset_arr = ['PhysionetMI']
    #dataset_arr = ['BNCI2014004']

    for dataset in dataset_arr:
        #for percentage in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        #for percentage in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        for approach in ['LDA']:##'CE_stSENet']:# 'EEGWaveNet']:#, 'EEGNet']:
        #for approach in ['EEGNet']:
        #for approach in ['BEEGNet']:
            all_scores = []
            align = False

            #pre_model_arch = 'EEGNet'

            fusion = 'None'
            # fusion = 'feature'
            # fusion = 'Autoencoder'
            # fusion = 'decision'
            # fusion = 'middle'
            # fusion = 'lossaid'
            # fusion = 'deepinversion'

            print(dataset, align, approach, fusion)
            #print(percentage)

            info = dataset_to_file(dataset, data_save=False)
            #info = None

            # eeg_handfeature(dataset, info, align, approach, cuda_device_id)
            eeg_ml(dataset, info, align, approach, cuda_device_id)
            #eeg_dnn(dataset, info, align, approach, cuda_device_id)
            #eeg_ml(dataset, info, align, approach, cuda_device_id, percentage)
            #eeg_dnn(dataset, info, align, approach, cuda_device_id, percentage)
            # eeg_dnn_ms(dataset, info, align, approach, cuda_device_id)
            # eeg_fusion(dataset, info, fusion, align, approach, pre_model_arch, cuda_device_id)
            # eeg_SFN(dataset, info, fusion, align, approach, pre_model_arch, cuda_device_id)

            # eeg_cotrain(dataset, info, align, approach, cuda_device_id)
