import sys
import random

import numpy as np
import mne
import torch
import os

from utils.data_utils import data_loader_split, convert_label, data_loader_DEAP, dataset_SEED_extracted_process, split_data, dataset_to_file, time_cut, feature_smooth_moving_average
from utils.alg_utils import EA
from nn_baseline import nn_fixepoch
from models.FC import FC
from models.EEGNet import EEGNet
from mne.decoding import CSP
from mne.preprocessing import Xdawn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC, LinearSVC
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, roc_auc_score, balanced_accuracy_score
from sklearn import preprocessing
from pyriemann.estimation import XdawnCovariances
from pyriemann.tangentspace import TangentSpace
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from pyriemann.tangentspace import TangentSpace
from torch.utils.data import TensorDataset, DataLoader
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from models.Autoencoder import Autoencoder, Autoencoder_encoder

from mne.decoding import CSP
from mne.preprocessing import Xdawn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC, LinearSVC
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, roc_auc_score, balanced_accuracy_score
from sklearn import preprocessing
from pyriemann.estimation import XdawnCovariances
from pyriemann.tangentspace import TangentSpace
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from pyriemann.tangentspace import TangentSpace


def apply_pca(train_x, test_x, variance_retained):
    #train_x_len = len(train_x)
    pca = PCA(variance_retained)
    #all_x = np.concatenate([train_x, test_x], axis=0)
    print('before PCA:', train_x.shape, test_x.shape)
    train_x = pca.fit_transform(train_x)
    test_x = pca.transform(test_x)
    print('after PCA:', train_x.shape, test_x.shape)
    print('PCA variance retained:', np.sum(pca.explained_variance_ratio_))
    #train_x = all_x_pca[:train_x_len, :]
    #test_x = all_x_pca[train_x_len:, :]
    return train_x, test_x


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


def apply_smote(train_x, train_y):
    smote = SMOTE(n_jobs=8)
    print('before SMOTE:', train_x.shape, train_y.shape)
    train_x, train_y = smote.fit_resample(train_x, train_y)
    print('after SMOTE:', train_x.shape, train_y.shape)
    return train_x, train_y


def apply_randup(train_x, train_y):
    sampler = RandomOverSampler()
    print('before Random Upsampling:', train_x.shape, train_y.shape)
    train_x, train_y = sampler.fit_resample(train_x, train_y)
    print('after Random Upsampling:', train_x.shape, train_y.shape)
    return train_x, train_y

'''
def deep_DEAP():
    data_folder = '/Users/Riccardo/Workspace/HUST-BCI/data/DEAP/data_preprocessed_matlab'

    data, labels = data_loader_DEAP(data_folder)

    num_subjects = len(data)
    data_cctnt = np.concatenate(data, axis=0)
    labels_cctnt = np.concatenate(labels, axis=0)

    # pre-trial 3 seconds removal to get 60 seconds data
    data_cctnt = data_cctnt[:, :, -7680:]
    # print(data_cctnt[0, :2, :5])

    # split the 60 seconds data
    split_multiplier = 6
    data_split_arr = split_data(data_cctnt, axis=2, times=split_multiplier)
    data_split = np.concatenate(data_split_arr, axis=0)
    print(data_split.shape)
    # print(data_split[0, :2, :5])

    # the EA
    data_EA = EA(data_split)
    print(data_EA.shape)
    # print(data_EA[0, :2, :5])

    # process labels
    labels = np.repeat(labels, split_multiplier)
    print(labels.shape)
    # print(labels[:100])

    total_samples = len(labels)
    print('total_samples:', total_samples)

    model_name = 'EEGNet'
    dataset = 'DEAP'
    score_arr = []

    for i in range(1, num_subjects + 1):
        data_subjects = np.split(data_EA, indices_or_sections=num_subjects, axis=0)
        labels_subjects = np.split(labels, indices_or_sections=num_subjects, axis=0)
        test_x = data_subjects.pop(i - 1)
        test_y = labels_subjects.pop(i - 1)
        train_x = data_subjects
        train_y = labels_subjects

        print('test_subj:', i)
        score = nn_fixepoch(
            test_subj=i,
            learning_rate=0.001,
            num_iterations=100,
            cuda=False,
            seed=42,
            test=False,
            test_path='',
            dataset=dataset,
            model_name=model_name,
            train_x=train_x,
            train_y=train_y,
            test_x=test_x,
            test_y=test_y
        )
        print(round(score, 5))
        score_arr.append(round(score, 5))
    print(dataset, model_name)
    print(round(np.average(score_arr), 5))
'''


def data_loader(dataset, align):
    mne.set_log_level('warning')

    if dataset == 'SEED':
        if align:
            X = np.load('./data/' + dataset + '/X_DE_EA.npy')
            y = np.load('./data/' + dataset + '/labels_EA.npy')
        else:
            X = np.load('./data/' + dataset + '/X_DE.npy')
            y = np.load('./data/' + dataset + '/labels.npy')
        # provided extracted feature
        #X = np.load('./data/' + dataset + '/X_extracted.npy')
        #y = np.load('./data/' + dataset + '/labels_extracted.npy')
    elif dataset == 'SEED-V':
        '''
        if align:
            X = np.load('./data/' + dataset + '/X_DE_EA_4sec.npy')
            y = np.load('./data/' + dataset + '/labels_EA_4sec.npy')
        else:
            X = np.load('./data/' + dataset + '/X_DE_4sec.npy')
            y = np.load('./data/' + dataset + '/labels_4sec.npy')
        '''
        # provided extracted feature
        X = np.load('./data/' + dataset + '/X_extracted.npy')
        y = np.load('./data/' + dataset + '/labels_extracted.npy')

    elif dataset == 'DEAP':
        if align:
            X = np.load('./data/' + dataset + '/X_DE_EA.npy')
            y = np.load('./data/' + dataset + '/labels_EA.npy')
        else:
            X = np.load('./data/' + dataset + '/X_DE.npy')
            y = np.load('./data/' + dataset + '/labels.npy')
    elif dataset == 'MI1':
        data = np.load('./data/' + dataset + '/MI1.npz')
        X = data['data']
        X = X.reshape(-1, X.shape[2], X.shape[3])
        y = data['label']
        y = y.reshape(-1, )
    else:
        X = np.load('./data/' + dataset + '/X.npy')
        y = np.load('./data/' + dataset + '/labels.npy')
    print(X.shape, y.shape)

    if dataset == 'BNCI2014001':
        paradigm = 'MI'
        num_subjects = 9
        sample_rate = 250

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

        # only use session train, remove session test
        indices = []
        for i in range(num_subjects):
            indices.append(np.arange(100) + (160 * i))
        indices = np.concatenate(indices, axis=0)
        X = X[indices]
        y = y[indices]
    elif dataset == 'MI1':
        paradigm = 'MI'
        num_subjects = 7
        sample_rate = 100
    elif dataset == 'BNCI2015004':
        paradigm = 'MI'
        num_subjects = 9
        trials_num_arr = [160, 160, 160, 150, 160, 160, 150, 160, 160]

        # only use session train, remove session test
        indices = []
        for i in range(3):
            indices.append(np.arange(80) + (160 * i))
        indices.append(np.arange(80) + (160 * 3))
        for i in range(2):
            indices.append(np.arange(80) + (160 * i) + (160 * 4))
        indices.append(np.arange(80) + (160 * 6))
        for i in range(2):
            indices.append(np.arange(80) + (160 * i) + (160 * 7))
        indices = np.concatenate(indices, axis=0)
        X = X[indices]
        y = y[indices]
    elif dataset == 'SEED':
        paradigm = 'Emotion'
        num_subjects = 15
        sample_rate = -1

        # only use session 1
        indices = []
        for i in range(num_subjects):
            indices.append(np.arange(3394) + (3394 * i * 3)) # extracted
            #indices.append(np.arange(842) + (842 * i * 3)) # 4sec
        indices = np.concatenate(indices, axis=0)
        X = X[indices]
        y = y[indices]
    elif dataset == 'SEED-V':
        paradigm = 'Emotion'
        num_subjects = 16
        sample_rate = -1

        # only use session 1
        indices = []
        for i in range(num_subjects):
            indices.append(np.arange(681) + (1823 * i))  # extracted # 681 541 601
            #indices.append(np.arange(676) + (1809 * i))  # 4sec
            #indices.append(np.arange(2728) + (7314 * i))  # 1sec
        indices = np.concatenate(indices, axis=0)
        X = X[indices]
        y = y[indices]


    elif dataset == 'DEAP':
        paradigm = 'Emotion'
        num_subjects = 32
        sample_rate = -1

        # valence
        #y = y[:, 0]
        #y = convert_label(y, 0, 5)
        # arousal
        y = y[:, 1]
        y = convert_label(y, 0, 5)
        # dominance
        #y = y[:, 2]
        #y = convert_label(y, 0, 5)
        # liking
        #y = y[:, 3]
        #y = convert_label(y, 0, 5)
        #np.set_printoptions(threshold=sys.maxsize)
        #print(y)

    le = preprocessing.LabelEncoder()
    y = le.fit_transform(y)
    print('data shape:', X.shape, ' labels shape:', y.shape)
    return X, y, num_subjects, paradigm, sample_rate


def ml_classifier(approach, output_probability, train_x, train_y, test_x):
    if approach == 'LDA':
        clf = LinearDiscriminantAnalysis()
    elif approach == 'LR':
        clf = LogisticRegression(max_iter=1000)
    elif approach == 'AdaBoost':
        clf = AdaBoostClassifier()
    elif approach == 'GradientBoost':
        clf = GradientBoostingClassifier()
    # clf = LinearDiscriminantAnalysis()
    # clf = SVC()
    # clf = LinearSVC()
    # clf = KNeighborsClassifier()
    clf.fit(train_x, train_y)
    if output_probability:
        pred = clf.predict_proba(test_x)
    else:
        pred = clf.predict(test_x)
    return pred


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


def traintest_split_within_subject(dataset, X, y, num_subjects, test_subject_id):
    # [2728, 2176, 2410]
    data_subjects = np.split(X, indices_or_sections=num_subjects, axis=0)
    labels_subjects = np.split(y, indices_or_sections=num_subjects, axis=0)
    if dataset == 'SEED-V':

        data1_arr = []
        data2_arr = []
        data3_arr = []
        label1_arr = []
        label2_arr = []
        label3_arr = []
        for data_subject, label_subject in zip(data_subjects, labels_subjects):
            data1 = data_subject[:2728]
            data2 = data_subject[2728:2728 + 2176]
            data3 = data_subject[2728 + 2176:]
            data1_arr.append(data1)
            data2_arr.append(data2)
            data3_arr.append(data3)
            label1 = label_subject[:2728]
            label2 = label_subject[2728:2728 + 2176]
            label3 = label_subject[2728 + 2176:]
            label1_arr.append(label1)
            label2_arr.append(label2)
            label3_arr.append(label3)
        '''
        data1_arr = np.concatenate(data1_arr, axis=0)
        data2_arr = np.concatenate(data2_arr, axis=0)
        data3_arr = np.concatenate(data3_arr, axis=0)
        label1_arr = np.concatenate(label1_arr, axis=0)
        label2_arr = np.concatenate(label2_arr, axis=0)
        label3_arr = np.concatenate(label3_arr, axis=0)
        '''
        data_arr = [data1_arr, data2_arr, data3_arr]
        label_arr = [label1_arr, label2_arr, label3_arr]
        return data_arr, label_arr


def data_alignment(X, num_subjects):
    # subject-wise EA
    print('before EA:', X.shape)
    out = []
    for i in range(num_subjects):
        tmp_x = EA(X[X.shape[0] // num_subjects * i:X.shape[0] // num_subjects * (i + 1), :, :])
        out.append(tmp_x)
    X = np.concatenate(out, axis=0)
    print('after EA:', X.shape)
    return X


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


def data_loader_eye():
    X = np.load('./data/' + 'SEED-V' + '/X_eye_extracted.npy')
    y = np.load('./data/' + 'SEED-V' + '/labels_eye_extracted.npy')

    # only use session 1
    indices = []
    for i in range(16):
        indices.append(np.arange(681) + (1823 * i))  # extracted
        #indices.append(np.arange(676) + (1809 * i))  # 4sec
        #indices.append(np.arange(2728) + (7314 * i))  # 1sec
    indices = np.concatenate(indices, axis=0)
    X = X[indices]
    y = y[indices]

    le = preprocessing.LabelEncoder()
    y = le.fit_transform(y)
    print('data shape:', X.shape, ' labels shape:', y.shape)

    return X, y


def eeg_classification(dataset, info, align, approach, cuda_device_id):
    X, y, num_subjects, paradigm, sample_rate = data_loader(dataset, align)
    print('sample rate:', sample_rate)

    if align and dataset != 'SEED' and dataset != 'SEED-V' and dataset != 'DEAP':
        X = data_alignment(X, num_subjects)

    scores_arr = []
    std_arr = []
    for i in range(num_subjects):
        train_x, train_y, test_x, test_y = traintest_split_cross_subject(dataset, X, y, num_subjects, i)

        # eye
        #if dataset == 'SEED-V':
        #    X_eye, y_eye = data_loader_eye()
        #    train_x_eye, train_y_eye, test_x_eye, test_y_eye = traintest_split_cross_subject(dataset, X_eye, y_eye, num_subjects, i)

        '''
        data_arr, label_arr = traintest_split_within_subject(dataset, X, y, num_subjects, i)
        print(data_arr[0][i].shape)
        train_x = np.concatenate((data_arr[0][i], data_arr[1][i]))
        test_x = data_arr[2][i]
        train_y = np.concatenate((label_arr[0][i], label_arr[1][i]))
        test_y = label_arr[2][i]
        
        print('train_x, train_y, test_x, test_y.shape', train_x.shape, train_y.shape, test_x.shape, test_y.shape)
        '''

        ar_unique, cnts = np.unique(y, return_counts=True)
        print("labels:", ar_unique)
        print("Counts:", cnts)

        '''
        moving_average_window = 10

        # feature smoothing
        #train_x = feature_smooth_moving_average(train_x, moving_average_window, 14, 1, 15, [235, 233, 206, 238, 185, 195, 237, 216, 265, 237, 235, 233, 235, 238, 206])
        #train_y = train_y[:train_x.shape[0]]
        test_x = feature_smooth_moving_average(test_x, moving_average_window, 1, 1, 15, [235, 233, 206, 238, 185, 195, 237, 216, 265, 237, 235, 233, 235, 238, 206])
        test_y = test_y[:test_x.shape[0]]

        # test set shrink
        test_x = test_x[1::moving_average_window, :]
        test_y = test_y[:test_x.shape[0]]
        '''

        print('train data shape:', train_x.shape, ' labels shape:', train_y.shape)
        print('test data shape:', test_x.shape, ' labels shape:', test_y.shape)

        # z-score standardization
        print('applying z-score:', train_x.shape, ' labels shape:', train_y.shape)
        train_x, test_x = apply_zscore(train_x, test_x, num_subjects)

        #scaler = preprocessing.StandardScaler()
        #train_x = scaler.fit_transform(train_x)
        #scaler = preprocessing.StandardScaler()
        #test_x = scaler.fit_transform(test_x)

        # Upsampling
        #train_x_xdawn, train_y = apply_smote(train_x_xdawn, train_y)
        #train_x, train_y = apply_randup(train_x, train_y)

        if approach != 'FC':
            # classifier
            pred = classifier(approach, train_x, train_y, test_x)
            score = np.round(balanced_accuracy_score(test_y, pred), 5)
        else:
            # deep model
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
                valid_percentage=0,
                label_probs=False,
                train_x=train_x,
                train_y=train_y,
                test_x=test_x,
                test_y=test_y)
        print('bca:', score)
        scores_arr.append(score)
    print('#' * 40)
    for i in range(len(scores_arr)):
        scores_arr[i] *= 100
    print('sbj scores', scores_arr)
    print('avg', np.round(np.average(scores_arr), 5))
    #print('std', np.round(np.std(scores_arr), 5))

    for i in range(len(std_arr)):
        std_arr[i] *= 100
    print('sbj stds', std_arr)
    print('std_randinit', np.round(np.average(std_arr), 5))


def autoencoder_model(cuda_device_id, train_x, test_x):
    num_iterations = 500
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

    model = Autoencoder(encoder_neurons=[train_x.shape[1], 32],
                        decoder_neurons=[32, train_x.shape[1]])
    model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()

    # Train the model
    for epoch in range(num_iterations):
        total_loss = 0
        cnt = 0
        for i, x in enumerate(train_loader):
            # Forward pass
            x = x[0].to(device)
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
    model_encoder = Autoencoder_encoder(encoder_neurons=[train_x.shape[1], 32])
    model_encoder_dict = model_encoder.state_dict()
    model_encoder.eval()
    model_encoder.to(device)

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
        x = x[0].to(device)
        outputs = model_encoder(x)
        outputs = outputs.cpu().detach().numpy()
        train_x_transformed.append(outputs)
    for i, x in enumerate(test_loader):
        x = x[0].to(device)
        outputs = model_encoder(x)
        outputs = outputs.cpu().detach().numpy()
        test_x_transformed.append(outputs)

    train_x_transformed = np.concatenate(train_x_transformed, axis=0)
    test_x_transformed = np.concatenate(test_x_transformed, axis=0)
    print('Autoencoder transformed:', train_x_transformed.shape, test_x_transformed.shape)

    return train_x_transformed, test_x_transformed


def load_and_predict_from_model(dataset, cuda_device_id, test_subj, test_x, arch_name):
    model_dir = './runs/' + dataset + '/'

    device = torch.device('cpu')
    if cuda_device_id:
        torch.cuda.manual_seed(seed)
        device = torch.device('cuda:' + str(cuda_device_id))
        print('using cuda...')

    model_path = model_dir + arch_name + '_testsubjID_' + str(test_subj) + '_epoch_100.ckpt'
    model = torch.load(model_path)
    model.to(device)
    model.eval()

    tensor_test_x = torch.from_numpy(test_x).to(torch.float32)
    test_dataset = TensorDataset(tensor_test_x)
    test_loader = DataLoader(test_dataset, batch_size=256)

    predicted_prob = []
    m = torch.nn.Softmax(dim=1)
    for i, x in enumerate(test_loader):
        x = x[0].to(device)
        outputs_prob = m(model(x))
        outputs_prob = outputs_prob.cpu().detach().numpy()

        predicted_prob.append(outputs_prob)
    predicted_prob = np.concatenate(predicted_prob, axis=0)
    return predicted_prob


def eeg_fusion(dataset, info, align, approach, cuda_device_id):
    X, y, num_subjects, paradigm, sample_rate = data_loader(dataset, align)
    print('sample rate:', sample_rate)

    if align and dataset != 'SEED' and dataset != 'SEED-V' and dataset != 'DEAP':
        X = data_alignment(X, num_subjects)

    scores_arr = []
    std_arr = []
    for i in range(num_subjects):
        X_eye, y_eye = data_loader_eye()
        train_x_eye, train_y_eye, test_x_eye, test_y_eye = traintest_split_cross_subject(dataset, X_eye, y_eye, num_subjects, i)

        train_x, train_y, test_x, test_y = traintest_split_cross_subject(dataset, X, y, num_subjects, i)


        print('train_x, train_y, test_x, test_y.shape', train_x.shape, train_y.shape, test_x.shape, test_y.shape)

        ar_unique, cnts = np.unique(y, return_counts=True)
        print("labels:", ar_unique)
        print("Counts:", cnts)


        print('train data shape:', train_x.shape, ' labels shape:', train_y.shape)
        print('test data shape:', test_x.shape, ' labels shape:', test_y.shape)

        # z-score standardization
        print('applying z-score eeg:', train_x.shape, ' labels shape:', train_y.shape)
        train_x, test_x = apply_zscore(train_x, test_x, num_subjects)
        #print('applying z-score eye:', train_x_eye.shape, ' labels shape:', train_y_eye.shape)
        #train_x_eye, test_x_eye = apply_zscore(train_x_eye, test_x_eye, num_subjects)
        '''
        pred_probs_eeg = load_and_predict_from_model(dataset, cuda_device_id, i, test_x, 'FC_eeg')
        pred_probs_eye = load_and_predict_from_model(dataset, cuda_device_id, i, test_x_eye, 'FC_eye')
        print(pred_probs_eeg, pred_probs_eye)
        pred_probs_ensemble = pred_probs_eeg + pred_probs_eye
        pred = np.argmax(pred_probs_ensemble, axis=1)
        score = np.round(accuracy_score(test_y, pred), 5)

        print('bca:', score)
        scores_arr.append(score)
        continue
        '''
        train_x = np.concatenate([train_x, train_x_eye], axis=1)
        test_x = np.concatenate([test_x, test_x_eye], axis=1)
        print('after feature fusion:', train_x.shape, test_x.shape)

        #if fusion == 'Autoencoder':
        train_x, test_x = autoencoder_model(cuda_device_id, train_x, test_x)

        #scaler = preprocessing.StandardScaler()
        #train_x = scaler.fit_transform(train_x)
        #scaler = preprocessing.StandardScaler()
        #test_x = scaler.fit_transform(test_x)

        # Upsampling
        #train_x_xdawn, train_y = apply_smote(train_x_xdawn, train_y)
        #train_x, train_y = apply_randup(train_x, train_y)

        if approach != 'FC':
            # classifier
            pred = classifier(approach, train_x, train_y, test_x)
            score = np.round(balanced_accuracy_score(test_y, pred), 5)
        else:
            # deep model
            feature_in = train_x.shape[1]
            class_out = len(np.unique(y))
            score = nn_fixepoch(model=FC(feature_in=feature_in, class_out=class_out),
                learning_rate=0.001,
                num_iterations=100,
                metrics=balanced_accuracy_score,
                cuda=True,
                cuda_device_id=cuda_device_id,
                seed=42,
                dataset=dataset,
                model_name='FC_autoencoder',
                test_subj_id=i,
                valid_percentage=0,
                label_probs=False,
                train_x=train_x,
                train_y=train_y,
                test_x=test_x,
                test_y=test_y)

        print('bca:', score)
        scores_arr.append(score)
    print('#' * 40)
    for i in range(len(scores_arr)):
        scores_arr[i] *= 100
    print('sbj scores', scores_arr)
    print('avg', np.round(np.average(scores_arr), 5))
    #print('std', np.round(np.std(scores_arr), 5))

    for i in range(len(std_arr)):
        std_arr[i] *= 100
    print('sbj stds', std_arr)
    print('std_randinit', np.round(np.average(std_arr), 5))


def dataset_SEEDV(data_folder):
    # 1000 Hz
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
                continue
                start_second = start_second2
                end_second = end_second2
                print('session 2')
            elif '_3_' in f_path:
                continue
                start_second = start_second3
                end_second = end_second3
                print('session 3')

            for j in range(15):

                data = data_mat[:, start_second[j] * 1000: end_second[j] * 1000]
                # print(data.shape)

                info = mne.create_info(ch_names=ch_names, sfreq=sample_freq, ch_types=['eeg'] * 62)
                raw = mne.io.RawArray(data, info)
                data = raw.get_data()
                data = mne.filter.resample(data, down=5) # 200 fs

                a = data.shape[-1] % 800  # 4 seconds chunks
                if data.shape[-1] % 800 != 0.0:
                    data = data[:, :-1 * a]
                data = split_data(data, axis=-1, times=data.shape[-1] // 800)
                data = np.stack(data, axis=0)
                print(data.shape)

                label = label_arr[j]
                label = np.full((data.shape[0], 1), label)

                subject_data.append(data)
                subject_labels.append(label)
            subject_data = np.concatenate(subject_data, axis=0)
            subject_labels = np.concatenate(subject_labels, axis=0).reshape(-1, )
            print('subject/session data shape:', subject_data.shape, subject_labels.shape)  # (681, 62, 4000) (681,)

            X.append(subject_data)
            labels.append(subject_labels)
            subj_trials_num.append(len(subject_labels))

    X = np.concatenate(X, axis=0)
    labels = np.concatenate(labels, axis=0).reshape(-1, )
    labels = labels.reshape(-1, )
    subj_trials_num = np.array(subj_trials_num)
    print(X.shape)  # (152730, 310)
    print(labels.shape)  # (152730,)
    print(subj_trials_num)  # 3394


    return X, labels, 16, 'emotion', 200


def eeg_dnn(dataset, info, align, approach, cuda_device_id):
    #X, y, num_subjects, paradigm, sample_rate = data_loader(dataset)
    X, y, num_subjects, paradigm, sample_rate = dataset_SEEDV('/mnt/data2/sylyoung/EEG/SEED-V/SEED-V/EEG_raw')

    print('sample rate:', sample_rate)

    unaligned_X = X.copy()

    if align:
        X = data_alignment(X, num_subjects)

    scores_arr = []
    std_arr = []
    for i in range(num_subjects):
        #train_x, train_y, test_x, test_y = traintest_split_within_subject(dataset, X, y, num_subjects, i, 0.5, True)

        train_x, train_y, test_x, test_y = traintest_split_cross_subject(dataset, X, y, num_subjects, i)

        metrics = accuracy_score

        class_out = len(np.unique(y))

        seed_arr = np.arange(1)
        rand_init_scores = []
        for seed in seed_arr:
            model = EEGNet(n_classes=class_out,
                           Chans=train_x.shape[1],
                           Samples=train_x.shape[2],
                           kernLenght=int(sample_rate // 2),
                           F1=4,
                           D=2,
                           F2=8,
                           dropoutRate=0.25,
                           norm_rate=0.5)
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
                                          test_y=test_y,)
                                          #loss_weights=loss_weights)
            rand_init_scores.append(rand_init_score)
        print('subj rand_init_scores:', rand_init_scores)
        score = np.round(np.average(rand_init_scores), 5)
        std = np.round(np.std(rand_init_scores), 5)
        scores_arr.append(score)
        std_arr.append(std)
        print('acc:', score, ', std:', std)

    print('#' * 40)
    for i in range(len(scores_arr)):
        scores_arr[i] *= 100
    print('sbj scores', scores_arr)
    print('avg', np.round(np.average(scores_arr), 5))

    for i in range(len(std_arr)):
        std_arr[i] *= 100
    print('sbj stds', std_arr)
    print('std_randinit', np.round(np.average(std_arr), 5))


if __name__ == '__main__':
    #dataset = 'BNCI2014001'
    #dataset = 'BNCI2014002'
    #dataset = 'MI1'
    #dataset = 'BNCI2015004'
    #dataset = 'BNCI2014008'
    #dataset = 'BNCI2014009'
    #dataset = 'BNCI2015003'
    #dataset = 'ERN'
    #dataset = 'SEED'
    dataset = 'SEED-V'
    #dataset = 'DEAP'

    cuda_device_id = 3
    cuda_device_id = str(cuda_device_id)

    align = True

    approach = 'EEGNet'

    print(dataset, align, approach)

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    # torch.manual_seed(seed)

    if dataset == 'SEED' or dataset == 'SEED-V' or dataset == 'DEAP':
        info = None
    else:
        info = dataset_to_file(dataset)

    #eeg_classification(dataset, info, align, approach, cuda_device_id)
    eeg_dnn(dataset, info, align, approach, cuda_device_id)
    #eeg_fusion(dataset, info, align, approach, cuda_device_id)
