import os.path as osp
import os
import math
import numpy as np
import random
import sys
import pandas as pd
import csv
import sklearn.metrics.pairwise
import torch as tr
import torch.nn as nn
import torch.utils.data
import torch.utils.data as Data
import moabb
import mne
import copy
import time
import scipy
from torchsampler import ImbalancedDatasetSampler
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from sklearn.metrics.pairwise import cosine_distances
from sklearn import preprocessing
from utils.sml import Sml



def time_cut(data, cut_percentage):
    # Time Cutting: cut at a certain percentage of the time. data: (..., ..., time_samples)
    data = data[:, :, :int(data.shape[2] * cut_percentage)]
    return data


def data_loader(dataset):
    '''

    :param dataset: str, dataset name
    :return: X, y, num_subjects, paradigm, sample_rate
    '''

    if dataset == 'MI1':
        data = np.load('./data/' + dataset + '/MI1.npz')
        X = data['data']
        X = X.reshape(-1, X.shape[2], X.shape[3])
        y = data['label']
        y = y.reshape(-1, )
    elif dataset == 'BNCI2014001-4':
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
        num_subjects = 105
        sample_rate = 160
        ch_num = 64

    le = preprocessing.LabelEncoder()
    y = le.fit_transform(y)
    print('data shape:', X.shape, ' labels shape:', y.shape)
    return X, y, num_subjects, paradigm, sample_rate, ch_num


def convert_label(labels, axis, threshold, minus1=False):
    if minus1:
        # Converting labels to -1 or 1, based on a certain threshold
        if np.random.randint(2, size=1)[0] == 1:
            label_01 = np.where(labels > threshold, 1, -1)
        else:
            label_01 = np.where(labels >= threshold, 1, -1)
    else:
        # Converting labels to 0 or 1, based on a certain threshold
        if np.random.randint(2, size=1)[0] == 1:
            label_01 = np.where(labels > threshold, 1, 0)
        else:
            label_01 = np.where(labels >= threshold, 1, 0)
    return label_01

def reverse_label(labels):
    # Reversing labels from 0 to 1, or 1 to 0
    return 1 - labels


def SML(preds):
    preds = convert_label(preds, 1, 0.5)
    hard = torch.from_numpy(preds).to(torch.float32)
    out = torch.mm(hard, hard.T)
    #out = np.cov(preds)
    w, v = np.linalg.eig(out)
    accuracies = v[:, 0]
    total = np.sum(accuracies)
    weights = accuracies / total
    prediction = np.dot(weights, hard)
    #prediction = np.dot(weights, preds)
    pred = convert_label(prediction, 0, 0.5)
    return pred


def SML_pred(preds):
    start_time = time.time()
    soft = torch.from_numpy(preds).to(torch.float32)
    out = torch.mm(soft, soft.T)
    #out = np.cov(preds)
    w, v = np.linalg.eig(out)
    accuracies = v[:, 0]
    total = np.sum(accuracies)
    weights = accuracies / total
    prediction = np.dot(weights, soft.numpy())
    #prediction = np.dot(weights, preds)
    pred = convert_label(prediction, 0, 0.5)
    SML_time = time.time()
    print('SML finished time in ms:', np.round((SML_time - start_time) * 1000,3))
    input('')

    return pred


def SML_pred_multiclass(preds, class_num):
    # pred (num_classifiers, num_samples, num_classes)
    predictions = []
    for i in range(class_num):
        soft = torch.from_numpy(preds[:, :, i]).to(torch.float32)
        out = torch.mm(soft, soft.T)
        w, v = np.linalg.eig(out)
        accuracies = v[:, 0]
        total = np.sum(accuracies)
        weights = accuracies / total
        prediction = np.dot(weights, soft.numpy())
        predictions.append(prediction)
    predictions = np.array(predictions)
    pred = np.argmax(predictions, axis=0)
    return pred


def voting_ensemble_binary(preds):
    # preds of numpy array shape (n_classifier, n_samples), predictions are 0/1
    n_classifier, n_samples = preds.shape
    sum_votes = np.sum(preds, axis=0)
    vote_pred = convert_label(sum_votes, 0, n_classifier / 2)
    return vote_pred

def voting_ensemble_multiclass(preds, n_classes):
    # preds of numpy array shape (n_classifier, n_samples)
    n_classifier, n_samples = preds.shape
    votes_mat = np.zeros((n_classes, n_samples))
    for i in range(n_classifier):
        for j in range(n_samples):
            class_id = preds[i, j]
            votes_mat[class_id, j] += 1
    votes_pred = []
    for i in range(n_samples):
        pred = np.random.choice(np.flatnonzero(votes_mat[:, i] == votes_mat[:, i].max()))
        votes_pred.append(pred)
    votes_pred = np.array(votes_pred)
    return votes_pred


def fix_random_seed(SEED):
    tr.manual_seed(SEED)
    tr.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)


def binary_classification():
    method = 'TTA-IM-pred'
    data_name_list = ['BNCI2014001', 'BNCI2014002', 'BNCI2015001']

    for data_name in data_name_list:

        print(data_name)

        X, y, num_subjects, paradigm, sample_rate, ch_num = data_loader(data_name)

        total_mean = [[], [], [], []]
        # total_mean = [[], [], [], [], [], []]

        if data_name == 'BNCI2014001': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 9, 22, 2, 1001, 250, 144, 248
        if data_name == 'BNCI2014002': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 14, 15, 2, 2561, 512, 100, 640
        if data_name == 'MI1': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 7, 59, 2, 300, 200, 100, 72
        if data_name == 'BNCI2015001': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 12, 13, 2, 2561, 512, 200, 640
        if data_name == 'BNCI2014008': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'ERP', 8, 8, 2, 206, 256, 4200, 48
        if data_name == 'BNCI2014009': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'ERP', 10, 16, 2, 206, 256, 1728, 48
        if data_name == 'BNCI2015003': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'ERP', 10, 8, 2, 206, 256, 2520, 48
        if data_name == 'BNCI2014001-4': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 9, 22, 4, 1001, 250, 288, 248

        seed_arr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        preds = []
        for SEED in seed_arr:
            path = './logs/' + str(data_name) + str(method) + '_seed_' + str(SEED) + "_pred.csv"
            df = pd.read_csv(path, header=None)
            preds.append(df.to_numpy())
        '''
        seed_arr = [1, 2, 3, 4, 5]  # EEGNet + ShallowCNN (5 + 5)
        preds = []
        for SEED in seed_arr:
            for meth in ['TTA-IM-pred', 'TTA-IM-ShallowCNN']:
                path = './logs/' + str(data_name) + str(meth) + '_seed_' + str(SEED) + "_pred.csv"
                df = pd.read_csv(path, header=None)
                preds.append(df.to_numpy())

        # EEGNet + ShallowCNN (5 + 5) + remaining one
        seed_arr = [6]
        for SEED in seed_arr:
            path = './logs/' + str(data_name) + str(method) + '_seed_' + str(SEED) + "_pred.csv"
            df = pd.read_csv(path, header=None)
            preds.append(df.to_numpy())
        '''
        preds = np.stack(preds)  # (num_models, num_subjects, num_test_samples)
        preds = preds[:, :, preds.shape[-1] // 2:]  # use only second half
        print('test set preds shape:', preds.shape)

        '''
        train_preds = []
        for SEED in seed_arr:
            seed_preds = []
            for target_subject_id in range(N):
                path = './logs/' + 'SML-train/' + str(data_name) + 'SML-train' + '_seed_' + str(SEED) + '_testsubj_' + str(target_subject_id) + "_pred.csv"
                df = pd.read_csv(path, header=None)
                seed_preds.append(df.to_numpy())
            train_preds.append(seed_preds)  # (num_subjects, num_test_samples)
        train_preds = np.stack(train_preds)  # (num_models, num_subjects, num_test_samples)
        train_preds = train_preds.reshape(len(seed_arr), N, -1)  # (num_models, num_subjects, num_train_samples)
        print('training set preds shape:', train_preds.shape)
        '''
        all_avg = []

        '''
        for subj in range(num_subjects):
            pred = preds[:, subj, :]
            true = y[np.arange(trial_num).astype(int) + trial_num * subj]
            seed_acc = []
            for s in range(11):
                seed_acc = []
                ens_prediction = convert_label(pred[s], 0, 0.5)
                ens_score = accuracy_score(true[len(true) // 2:], ens_prediction[len(true) // 2:])
                seed_acc.append(ens_score)
            all_avg.append(np.average(seed_acc))
        all_avg = np.average(all_avg)
        print(all_avg)
        '''

        for ens_num in range(10, 11):

            print('Ensembling of ' + str(ens_num) + ' models...')

            acc_avg = []
            acc_vote = []
            acc_sml = []
            acc_smlpred = []
            # acc_sml_train = []
            # acc_smlpred_train = []

            for subj in range(num_subjects):
                '''
                # remove 4 bad subjects from each dataset
                if data_name == 'BNCI2014001':
                    if subj in [1, 4, 5, 6]:
                        continue
                if data_name == 'BNCI2014002':
                    if subj in [0, 7, 12, 13]:
                        continue
                if data_name == 'BNCI2015001':
                    if subj in [7, 8, 10, 11]:
                        continue
                print('removing subjects with bad performances...')
                '''
                pred = preds[:, subj, :]
                # train_pred = train_preds[:, subj, :]
                true = y[np.arange(trial_num).astype(int) + trial_num * subj]
                test_trial_num = trial_num
                true = y[np.arange(trial_num // 2).astype(int) + trial_num * subj + trial_num // 2]  # use second half
                test_trial_num = trial_num // 2  # use second half

                # average
                seed_acc = []
                for i in range(10):
                    ens_ids = np.arange(ens_num).astype(int) + i + 1
                    for k in range(len(ens_ids)):
                        if ens_ids[k] >= 11:
                            ens_ids[k] -= 11
                    ens_pred = np.average(pred[ens_ids, :], axis=0)
                    ens_prediction = convert_label(ens_pred, 0, 0.5)
                    ens_score = accuracy_score(true, ens_prediction)
                    seed_acc.append(ens_score)
                acc_avg.append(seed_acc)

                # voting
                seed_acc = []
                for i in range(10):
                    ens_ids = np.arange(ens_num).astype(int) + i + 1
                    for k in range(len(ens_ids)):
                        if ens_ids[k] >= 11:
                            ens_ids[k] -= 11
                    votes = []
                    for id_ in ens_ids:
                        vote_single = convert_label(pred[id_, :], 0, 0.5)
                        votes.append(vote_single)
                    votes = np.stack(votes)
                    ens_prediction = voting_ensemble_multiclass(votes, n_classes=class_num)
                    ens_score = accuracy_score(true, ens_prediction)
                    seed_acc.append(ens_score)
                acc_vote.append(seed_acc)

                # SML (TTA)
                seed_acc = []
                for i in range(10):
                    ens_ids = np.arange(ens_num).astype(int) + i + 1
                    for k in range(len(ens_ids)):
                        if ens_ids[k] >= 11:
                            ens_ids[k] -= 11
                    ens_prediction = []
                    for sample in range(test_trial_num):
                        # if sample < ens_num or ens_num <= 2:
                        if sample < ens_num:
                            ens_pred = np.average(pred[ens_ids, sample], axis=0)
                            curr_pred = convert_label(ens_pred, 0, 0.5).item()
                        else:
                            curr_table = pred[ens_ids, :sample + 1]
                            curr_pred = SML(curr_table)[-1]
                            '''
                            curr_table = convert_label(curr_table, 1, 0.5, minus1=True).astype(int)
                            sml = Sml(prevalence=0.5)
                            sml.fit(curr_table, tol=1e-3, max_iter=5000)
                            curr_pred = sml.get_inference(curr_table)[-1]
                            '''
                        ens_prediction.append(curr_pred)
                    ens_score = accuracy_score(true, ens_prediction)
                    seed_acc.append(ens_score)
                acc_sml.append(seed_acc)

                # SML pred (TTA)
                seed_acc = []
                for i in range(10):
                    ens_ids = np.arange(ens_num).astype(int) + i + 1
                    for k in range(len(ens_ids)):
                        if ens_ids[k] >= 11:
                            ens_ids[k] -= 11
                    ens_prediction = []
                    for sample in range(test_trial_num):
                        if sample < ens_num:
                            ens_pred = np.average(pred[ens_ids, sample], axis=0)
                            curr_pred = convert_label(ens_pred, 0, 0.5).item()
                        else:
                            curr_table = pred[ens_ids, :sample + 1]
                            curr_pred = SML_pred(curr_table)[-1]
                        ens_prediction.append(curr_pred)
                    ens_score = accuracy_score(true, ens_prediction)
                    seed_acc.append(ens_score)
                acc_smlpred.append(seed_acc)
                """
                # SML with training set (in TTA)
                seed_acc = []
                for i in range(10):
                    ens_ids = np.arange(ens_num).astype(int) + i + 1
                    for k in range(len(ens_ids)):
                        if ens_ids[k] >= 11:
                            ens_ids[k] -= 11
                    ens_prediction = []
                    for sample in range(test_trial_num):
                        #if ens_num <= 2:
                        if False:
                            ens_pred = np.average(pred[ens_ids, sample], axis=0)
                            curr_pred = convert_label(ens_pred, 0, 0.5).item()
                        else:
                            te_pred = pred[ens_ids, :sample + 1]
                            tr_pred = train_pred[ens_ids, :]
                            curr_table = np.concatenate((tr_pred, te_pred), axis=1)
                            curr_pred = SML(curr_table)[-1]
                            '''
                            curr_table = convert_label(curr_table, 1, 0.5, minus1=True).astype(int)
                            sml = Sml(prevalence=0.5)
                            sml.fit(curr_table, tol=1e-3, max_iter=5000)
                            curr_pred = sml.get_inference(curr_table)[-1]
                            '''
                        ens_prediction.append(curr_pred)
                    ens_score = accuracy_score(true, ens_prediction)
                    seed_acc.append(ens_score)
                acc_sml_train.append(seed_acc)

                # SML pred with training set (in TTA)
                seed_acc = []
                for i in range(10):
                    ens_ids = np.arange(ens_num).astype(int) + i + 1
                    for k in range(len(ens_ids)):
                        if ens_ids[k] >= 11:
                            ens_ids[k] -= 11
                    ens_prediction = []
                    for sample in range(test_trial_num):
                        te_pred = pred[ens_ids, :sample + 1]
                        tr_pred = train_pred[ens_ids, :]
                        curr_table = np.concatenate((tr_pred, te_pred), axis=1)
                        curr_pred = SML_pred(curr_table)[-1]
                        ens_prediction.append(curr_pred)
                    ens_score = accuracy_score(true, ens_prediction)
                    seed_acc.append(ens_score)
                acc_smlpred_train.append(seed_acc)
                """

            method_cnt = 0
            # for score in [acc_avg, acc_vote, acc_sml, acc_smlpred, acc_sml_train, acc_smlpred_train]:
            for score in [acc_avg, acc_vote, acc_sml, acc_smlpred]:

                if method_cnt == 0:
                    print('###############Average Ensemble###############')
                if method_cnt == 1:
                    print('###############Voting Ensemble################')
                if method_cnt == 2:
                    print('###############SML Ensemble###################')
                if method_cnt == 3:
                    print('###############SMLpred Ensemble###############')
                # if method_cnt == 4:
                #    print('###############SMLwithtrain Ensemble###############')
                # if method_cnt == 5:
                #    print('###############SMLpredwithtrain Ensemble###############')

                score = np.array(score).transpose((1, 0))
                subject_mean = np.round(np.average(score, axis=0) * 100, 2)
                dataset_mean = np.round(np.average(np.average(score)) * 100, 2)
                dataset_std = np.round(np.std(np.average(score, axis=1)) * 100, 2)

                print(subject_mean)
                print(dataset_mean)
                print(dataset_std)

                total_mean[method_cnt].append(dataset_mean)

                method_cnt += 1

        print(total_mean)


def multiclass_classification():
    method = 'TTA-IM-4c-slide2'
    data_name_list = ['BNCI2014001-4']

    for data_name in data_name_list:

        print(data_name)

        X, y, num_subjects, paradigm, sample_rate, ch_num = data_loader(data_name)

        total_mean = [[], [], []]
        # total_mean = [[], [], [], [], [], []]

        if data_name == 'BNCI2014001': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 9, 22, 2, 1001, 250, 144, 248
        if data_name == 'BNCI2014002': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 14, 15, 2, 2561, 512, 100, 640
        if data_name == 'MI1': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 7, 59, 2, 300, 200, 100, 72
        if data_name == 'BNCI2015001': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 12, 13, 2, 2561, 512, 200, 640
        if data_name == 'BNCI2014008': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'ERP', 8, 8, 2, 206, 256, 4200, 48
        if data_name == 'BNCI2014009': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'ERP', 10, 16, 2, 206, 256, 1728, 48
        if data_name == 'BNCI2015003': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'ERP', 10, 8, 2, 206, 256, 2520, 48
        if data_name == 'BNCI2014001-4': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 9, 22, 4, 1001, 250, 288, 248

        seed_arr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        preds = []
        for SEED in seed_arr:
            path = './logs/' + str(data_name) + str(method) + '_seed_' + str(SEED) + "_pred.csv"
            df = pd.read_csv(path, header=None)
            preds.append(df.to_numpy())
        '''
        seed_arr = [1, 2, 3, 4, 5]  # EEGNet + ShallowCNN (5 + 5)
        preds = []
        for SEED in seed_arr:
            for meth in ['TTA-IM-pred', 'TTA-IM-ShallowCNN']:
                path = './logs/' + str(data_name) + str(meth) + '_seed_' + str(SEED) + "_pred.csv"
                df = pd.read_csv(path, header=None)
                preds.append(df.to_numpy())

        # EEGNet + ShallowCNN (5 + 5) + remaining one
        seed_arr = [6]
        for SEED in seed_arr:
            path = './logs/' + str(data_name) + str(method) + '_seed_' + str(SEED) + "_pred.csv"
            df = pd.read_csv(path, header=None)
            preds.append(df.to_numpy())
        '''
        preds = np.stack(preds)  # (num_models, num_subjects, num_test_samples)
        preds = preds.reshape(len(seed_arr), N, trial_num, class_num)
        # preds = preds[:, :, preds.shape[-1] // 2:]  # use only second half
        print('test set preds shape:', preds.shape)

        '''
        train_preds = []
        for SEED in seed_arr:
            seed_preds = []
            for target_subject_id in range(N):
                path = './logs/' + 'SML-train/' + str(data_name) + 'SML-train' + '_seed_' + str(SEED) + '_testsubj_' + str(target_subject_id) + "_pred.csv"
                df = pd.read_csv(path, header=None)
                seed_preds.append(df.to_numpy())
            train_preds.append(seed_preds)  # (num_subjects, num_test_samples)
        train_preds = np.stack(train_preds)  # (num_models, num_subjects, num_test_samples)
        train_preds = train_preds.reshape(len(seed_arr), N, -1)  # (num_models, num_subjects, num_train_samples)
        print('training set preds shape:', train_preds.shape)
        '''
        all_avg = []

        '''
        for subj in range(num_subjects):
            pred = preds[:, subj, :]
            true = y[np.arange(trial_num).astype(int) + trial_num * subj]
            seed_acc = []
            for s in range(11):
                seed_acc = []
                ens_prediction = convert_label(pred[s], 0, 0.5)
                ens_score = accuracy_score(true[len(true) // 2:], ens_prediction[len(true) // 2:])
                seed_acc.append(ens_score)
            all_avg.append(np.average(seed_acc))
        all_avg = np.average(all_avg)
        print(all_avg)
        '''

        for ens_num in range(2, 11):

            print('Ensembling of ' + str(ens_num) + ' models...')

            acc_avg = []
            acc_vote = []
            #acc_sml = []
            acc_smlpred = []
            # acc_sml_train = []
            # acc_smlpred_train = []

            for subj in range(num_subjects):
                '''
                # remove 4 bad subjects from each dataset
                if data_name == 'BNCI2014001':
                    if subj in [1, 4, 5, 6]:
                        continue
                if data_name == 'BNCI2014002':
                    if subj in [0, 7, 12, 13]:
                        continue
                if data_name == 'BNCI2015001':
                    if subj in [7, 8, 10, 11]:
                        continue
                print('removing subjects with bad performances...')
                '''
                pred = preds[:, subj, :, :]
                # train_pred = train_preds[:, subj, :]
                true = y[np.arange(trial_num).astype(int) + trial_num * subj]
                test_trial_num = trial_num
                # true = y[np.arange(trial_num // 2).astype(int) + trial_num * subj + trial_num // 2]  # use second half
                # test_trial_num = trial_num // 2  # use second half

                # average
                seed_acc = []
                for i in range(10):
                    ens_ids = np.arange(ens_num).astype(int) + i + 1
                    for k in range(len(ens_ids)):
                        if ens_ids[k] >= 11:
                            ens_ids[k] -= 11
                    ens_pred = np.average(pred[ens_ids], axis=0)
                    ens_pred = np.argmax(ens_pred, axis=-1)
                    ens_score = accuracy_score(true, ens_pred)
                    seed_acc.append(ens_score)
                acc_avg.append(seed_acc)

                # voting
                seed_acc = []
                for i in range(10):
                    ens_ids = np.arange(ens_num).astype(int) + i + 1
                    for k in range(len(ens_ids)):
                        if ens_ids[k] >= 11:
                            ens_ids[k] -= 11
                    votes = []
                    for id_ in ens_ids:
                        vote_single = np.argmax(pred[id_, :, :], axis=-1)
                        votes.append(vote_single)
                    votes = np.stack(votes)
                    ens_prediction = voting_ensemble_multiclass(votes, n_classes=class_num)
                    ens_score = accuracy_score(true, ens_prediction)
                    seed_acc.append(ens_score)
                acc_vote.append(seed_acc)
                """
                # SML (TTA)
                seed_acc = []
                for i in range(10):
                    ens_ids = np.arange(ens_num).astype(int) + i + 1
                    for k in range(len(ens_ids)):
                        if ens_ids[k] >= 11:
                            ens_ids[k] -= 11
                    ens_prediction = []
                    for sample in range(test_trial_num):
                        # if sample < ens_num or ens_num <= 2:
                        if sample < ens_num:
                            ens_pred = np.average(pred[ens_ids, sample], axis=0)
                            curr_pred = convert_label(ens_pred, 0, 0.5).item()
                        else:
                            curr_table = pred[ens_ids, :sample + 1]
                            curr_pred = SML(curr_table)[-1]
                            '''
                            curr_table = convert_label(curr_table, 1, 0.5, minus1=True).astype(int)
                            sml = Sml(prevalence=0.5)
                            sml.fit(curr_table, tol=1e-3, max_iter=5000)
                            curr_pred = sml.get_inference(curr_table)[-1]
                            '''
                        ens_prediction.append(curr_pred)
                    ens_score = accuracy_score(true, ens_prediction)
                    seed_acc.append(ens_score)
                acc_sml.append(seed_acc)
                '"""
                # SML pred (TTA)
                seed_acc = []
                for i in range(10):
                    ens_ids = np.arange(ens_num).astype(int) + i + 1
                    for k in range(len(ens_ids)):
                        if ens_ids[k] >= 11:
                            ens_ids[k] -= 11
                    ens_prediction = []
                    for sample in range(test_trial_num):
                        if sample < ens_num:
                            ens_pred = np.average(pred[ens_ids, sample, :], axis=0)
                            curr_pred = np.argmax(ens_pred, axis=-1)
                        else:
                            curr_table = pred[ens_ids, :sample + 1, :]
                            curr_pred = SML_pred_multiclass(curr_table, class_num)[-1]
                        ens_prediction.append(curr_pred)
                    ens_score = accuracy_score(true, ens_prediction)
                    seed_acc.append(ens_score)
                acc_smlpred.append(seed_acc)
                """
                # SML with training set (in TTA)
                seed_acc = []
                for i in range(10):
                    ens_ids = np.arange(ens_num).astype(int) + i + 1
                    for k in range(len(ens_ids)):
                        if ens_ids[k] >= 11:
                            ens_ids[k] -= 11
                    ens_prediction = []
                    for sample in range(test_trial_num):
                        #if ens_num <= 2:
                        if False:
                            ens_pred = np.average(pred[ens_ids, sample], axis=0)
                            curr_pred = convert_label(ens_pred, 0, 0.5).item()
                        else:
                            te_pred = pred[ens_ids, :sample + 1]
                            tr_pred = train_pred[ens_ids, :]
                            curr_table = np.concatenate((tr_pred, te_pred), axis=1)
                            curr_pred = SML(curr_table)[-1]
                            '''
                            curr_table = convert_label(curr_table, 1, 0.5, minus1=True).astype(int)
                            sml = Sml(prevalence=0.5)
                            sml.fit(curr_table, tol=1e-3, max_iter=5000)
                            curr_pred = sml.get_inference(curr_table)[-1]
                            '''
                        ens_prediction.append(curr_pred)
                    ens_score = accuracy_score(true, ens_prediction)
                    seed_acc.append(ens_score)
                acc_sml_train.append(seed_acc)

                # SML pred with training set (in TTA)
                seed_acc = []
                for i in range(10):
                    ens_ids = np.arange(ens_num).astype(int) + i + 1
                    for k in range(len(ens_ids)):
                        if ens_ids[k] >= 11:
                            ens_ids[k] -= 11
                    ens_prediction = []
                    for sample in range(test_trial_num):
                        te_pred = pred[ens_ids, :sample + 1]
                        tr_pred = train_pred[ens_ids, :]
                        curr_table = np.concatenate((tr_pred, te_pred), axis=1)
                        curr_pred = SML_pred(curr_table)[-1]
                        ens_prediction.append(curr_pred)
                    ens_score = accuracy_score(true, ens_prediction)
                    seed_acc.append(ens_score)
                acc_smlpred_train.append(seed_acc)
                """

            method_cnt = 0
            # for score in [acc_avg, acc_vote, acc_sml, acc_smlpred, acc_sml_train, acc_smlpred_train]:
            for score in [acc_avg, acc_vote, acc_smlpred]:

                if method_cnt == 0:
                    print('###############Average Ensemble###############')
                if method_cnt == 1:
                    print('###############Voting Ensemble################')
                if method_cnt == 2:
                    print('###############SMLpred Ensemble###############')
                # if method_cnt == 4:
                #    print('###############SMLwithtrain Ensemble###############')
                # if method_cnt == 5:
                #    print('###############SMLpredwithtrain Ensemble###############')

                score = np.array(score).transpose((1, 0))
                subject_mean = np.round(np.average(score, axis=0) * 100, 2)
                dataset_mean = np.round(np.average(np.average(score)) * 100, 2)
                dataset_std = np.round(np.std(np.average(score, axis=1)) * 100, 2)

                print(subject_mean)
                print(dataset_mean)
                print(dataset_std)

                total_mean[method_cnt].append(dataset_mean)

                method_cnt += 1

        print(total_mean)


if __name__ == '__main__':
    fix_random_seed(42)
    binary_classification()
    #multiclass_classification()


        
