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


def convert_label(labels, axis, threshold):
    # Converting labels to 0 or 1, based on a certain threshold
    if np.random.randint(2, size=1)[0] == 1:
        label_01 = np.where(labels > threshold, 1, 0)
    else:
        label_01 = np.where(labels >= threshold, 1, 0)
    # print(label_01)
    return label_01


def reverse_label(labels):
    # Reversing labels from 0 to 1, or 1 to 0
    return 1 - labels


def SML(preds):
    preds = convert_label(preds, 1, 0.5)
    hard = torch.from_numpy(preds).to(torch.float32)
    out = torch.mm(hard, hard.T)
    w, v = np.linalg.eig(out)
    accuracies = v[:, 0]
    total = np.sum(accuracies)
    weights = accuracies / total
    prediction = np.dot(weights, hard)
    pred = convert_label(prediction, 0, 0.5)
    return pred


def SML_pred(preds):
    soft = torch.from_numpy(preds).to(torch.float32)
    out = torch.mm(soft, soft.T)
    w, v = np.linalg.eig(out)
    accuracies = v[:, 0]
    total = np.sum(accuracies)
    weights = accuracies / total
    prediction = np.dot(weights, soft.numpy())
    pred = convert_label(prediction, 0, 0.5)
    return pred


def voting_ensemble(preds):
    # preds of numpy array shape (n_classifier, n_samples), predictions are 0/1
    n_classifier, n_samples = preds.shape
    sum_votes = np.sum(preds, axis=0)
    vote_pred = convert_label(sum_votes, 0, n_classifier / 2)
    return vote_pred


def fix_random_seed(SEED):
    tr.manual_seed(SEED)
    tr.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)


if __name__ == '__main__':
    fix_random_seed(42)

    method = 'TTA-IM-pred'
    data_name_list = ['BNCI2014001', 'BNCI2014002', 'BNCI2015001']

    seed_arr = [1, 2, 3, 4, 5]
    subj_num = [9, 14, 12]
    path = '/Users/Riccardo/Workspace/HUST-BCI/repos/EEG/logs/Imbsingle-TTA-IM-T-TIME-thresh.7-append4-AUC_pred.csv'
    all_preds_df = pd.read_csv(path, header=None, names=range(150))

    for data_name in data_name_list:
        cnt = 0

        X, y, num_subjects, paradigm, sample_rate, ch_num = data_loader(data_name)

        total_mean = [[], [], [], []]

        if data_name == 'BNCI2014001':
            trial_num = 144 - 36
        if data_name == 'BNCI2014002':
            trial_num = 100 - 25
        if data_name == 'BNCI2015001':
            trial_num = 200 - 50
        #print(sum(subj_num[:cnt]), sum(subj_num[:(cnt+1)]))
        #print(all_preds_df[0])
        preds = all_preds_df.loc[sum(subj_num[:cnt]):sum(subj_num[:(cnt+1)]), :]
        preds = preds.to_numpy()
        print(preds.shape)
        cnt += 1

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

        for ens_num in range(5, 6):
        #for ens_num in range(2, 11):

            print('Ensembling of ' + str(ens_num) + ' models...')

            acc_avg = []
            acc_vote = []
            acc_sml = []
            acc_smlpred = []

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
                '''
                pred = preds[subj, :]
                pred = pred[:trial_num]
                true = y[np.arange(trial_num).astype(int) + trial_num * subj]
                print('pred.shape:', pred.shape)

                # average
                seed_acc = []
                ens_ids = np.arange(ens_num).astype(int)
                ens_pred = np.average(pred[ens_ids], axis=0)
                ens_prediction = convert_label(ens_pred, 0, 0.5)
                ens_score = accuracy_score(true, ens_prediction)
                seed_acc.append(ens_score)
                acc_avg.append(seed_acc)

                # voting
                seed_acc = []
                ens_ids = np.arange(ens_num).astype(int)
                votes = []
                for id_ in ens_ids:
                    vote_single = convert_label(pred[id_, :], 0, 0.5)
                    votes.append(vote_single)
                votes = np.stack(votes)
                ens_prediction = voting_ensemble(votes)
                ens_score = accuracy_score(true, ens_prediction)
                seed_acc.append(ens_score)
                acc_vote.append(seed_acc)

                # SML (TTA)
                seed_acc = []
                ens_ids = np.arange(ens_num).astype(int)
                ens_prediction = []
                for sample in range(trial_num):
                    if sample <= ens_num:
                        ens_pred = np.average(pred[ens_ids, sample], axis=0)
                        curr_pred = convert_label(ens_pred, 0, 0.5).item()
                    else:
                        curr_table = pred[ens_ids, :sample + 1]
                        curr_pred = SML(curr_table)[-1]
                    ens_prediction.append(curr_pred)
                ens_score = accuracy_score(true, ens_prediction)
                seed_acc.append(ens_score)
                acc_sml.append(seed_acc)

                # SML pred (TTA)
                seed_acc = []
                ens_ids = np.arange(ens_num).astype(int)
                ens_prediction = []
                for sample in range(trial_num):
                    if sample <= ens_num:
                        ens_pred = np.average(pred[ens_ids, sample], axis=0)
                        curr_pred = convert_label(ens_pred, 0, 0.5).item()
                    else:
                        curr_table = pred[ens_ids, :sample + 1]
                        curr_pred = SML_pred(curr_table)[-1]
                    ens_prediction.append(curr_pred)
                ens_score = accuracy_score(true, ens_prediction)
                seed_acc.append(ens_score)
                acc_smlpred.append(seed_acc)

            method_cnt = 0
            for score in [acc_avg, acc_vote, acc_sml, acc_smlpred]:

                if method_cnt == 0:
                    print('###############Average Ensemble###############')
                if method_cnt == 1:
                    print('###############Voting Ensemble################')
                if method_cnt == 2:
                    print('###############SML Ensemble###################')
                if method_cnt == 3:
                    print('###############SMLpred Ensemble###############')

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

