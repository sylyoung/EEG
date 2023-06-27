import math
import os
import random
import time
import sys

import numpy as np
import mne
import torch
import torch.nn as nn
import sklearn
import scipy.io as sio
from torch.utils.data import TensorDataset, DataLoader


def nn_fixepoch(
        model,
        learning_rate,
        num_iterations,
        batch_size,
        metrics,
        cuda,
        cuda_device_id,
        seed,
        dataset,
        model_name,
        test_subj_id,
        label_probs,
        valid_percentage,
        train_x,
        train_y,
        test_x,
        test_y,
        loss_weights=None,
):
    start_time = time.time()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device('cpu')
    if cuda:
        torch.cuda.manual_seed(seed)
        device = torch.device('cuda:' + str(cuda_device_id))
        print('using cuda...')

    if valid_percentage != None and valid_percentage > 0:
        valid_indices = []
        for i in range(len(train_x) // 100):
            indices = np.arange(valid_percentage) + 100 * i
            valid_indices.append(indices)
        valid_indices = np.concatenate(valid_indices)
        valid_x = train_x[valid_indices]
        valid_y = train_y[valid_indices]
        train_x = np.delete(train_x, valid_indices, 0)
        train_y = np.delete(train_y, valid_indices, 0)

        print('train_x.shape, train_y.shape, valid_x.shape, valid_y.shape:', train_x.shape, train_y.shape, valid_x.shape, valid_y.shape)

    if label_probs:
        tensor_train_x, tensor_train_y = torch.from_numpy(train_x).to(
            torch.float32), torch.from_numpy(train_y).to(torch.float32)
    else:
        tensor_train_x, tensor_train_y = torch.from_numpy(train_x).to(
            torch.float32), torch.from_numpy(train_y.reshape(-1, )).to(torch.long)
    if model_name == 'EEGNet':
        tensor_train_x = tensor_train_x.unsqueeze_(3).permute(0, 3, 1, 2)
    if model_name == 'CE_stSENet':
        tensor_train_x = tensor_train_x.unsqueeze_(2)
    train_dataset = TensorDataset(tensor_train_x, tensor_train_y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size)

    if valid_percentage != None and valid_percentage > 0:
        if label_probs:
            tensor_valid_x, tensor_valid_y = torch.from_numpy(valid_x).to(
                torch.float32), torch.from_numpy(valid_y).to(torch.float32)
        else:
            tensor_valid_x, tensor_valid_y = torch.from_numpy(valid_x).to(
                torch.float32), torch.from_numpy(valid_y.reshape(-1,)).to(torch.long)
        if model_name == 'EEGNet':
            tensor_valid_x = tensor_valid_x.unsqueeze_(3).permute(0, 3, 1, 2)
        if model_name == 'CE_stSENet':
            tensor_valid_x = tensor_valid_x.unsqueeze_(2)
        valid_dataset = TensorDataset(tensor_valid_x, tensor_valid_y)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size)

    tensor_test_x, tensor_test_y = torch.from_numpy(test_x).to(
        torch.float32), torch.from_numpy(test_y.reshape(-1,)).to(torch.long)
    if model_name == 'EEGNet':
        tensor_test_x = tensor_test_x.unsqueeze_(3).permute(0, 3, 1, 2)
    if model_name == 'CE_stSENet':
        tensor_test_x = tensor_test_x.unsqueeze_(2)
    test_dataset = TensorDataset(tensor_test_x, tensor_test_y)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    if cuda:
        model.cuda()
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)

    criterion = torch.nn.CrossEntropyLoss()
    if loss_weights is not None:
        criterion = torch.nn.CrossEntropyLoss(weight=loss_weights)

    if not os.path.isdir('./runs/'):
        os.mkdir('./runs/')

    if not os.path.isdir('./runs/' + str(dataset) + '/'):
        os.mkdir('./runs/' + str(dataset))

    # Train the model
    for epoch in range(num_iterations):
        total_loss = 0
        cnt = 0
        for i, (x, y) in enumerate(train_loader):
            # Forward pass
            if cuda:
                x = x.cuda()
                y = y.cuda()

            outputs = model(x)

            loss = criterion(outputs, y)

            total_loss += loss
            cnt += 1

            # Backward and optimize
            opt.zero_grad()
            loss.backward()
            opt.step()
        out_loss = total_loss / cnt

        if (epoch + 1) % 1 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_iterations, out_loss))

        if (epoch + 1) % num_iterations == 0:
            torch.save(model, './runs/' + str(dataset) + '/' + model_name + '_EA_' + str(test_subj_id) +
                       '_epoch_' + str(epoch + 1) + '_seed_' + str(seed) + '.ckpt')
            print("--- %s seconds ---" % (time.time() - start_time))

    # Test the model
    if valid_percentage != None and valid_percentage > 0:
        model = torch.load('./runs/' + str(dataset) + '/' + model_name + '_' + str(test_subj_id) + '_best.ckpt')
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        all_output = []
        for x, y in test_loader:
            if cuda:
                x = x.cuda()
                y = y.cuda()
            outputs = model(x)
            all_output.append(outputs)
            _, predicted = torch.max(outputs.data, 1)
            y_true.append(y.cpu())
            y_pred.append(predicted.cpu())
        if metrics is None:
            all_output = torch.nn.Softmax(dim=1)(torch.cat(all_output))
            all_output = all_output.detach().cpu()
            predict = np.zeros(all_output.shape[0])
            for m in range(all_output.shape[0]):
                if all_output[m][1] >= 0.3:
                    predict[m] = 1.0
            all_label = np.concatenate(y_true).reshape(-1, )
            f1 = sklearn.metrics.f1_score(all_label, predict, average='weighted')  # 这个weighted一定要加
            bca = sklearn.metrics.balanced_accuracy_score(all_label, predict)
            tn, fp, fn, tp = sklearn.metrics.confusion_matrix(all_label, predict).ravel()
            acc = (tp + tn) / (tp + tn + fp + fn)
            sen = tp / (tp + fn)
            spec = tn / (tn + fp)
            all_label = np.eye(2)[all_label]
            try:
                auc = sklearn.metrics.roc_auc_score(all_label, all_output, average='weighted')
            except ValueError:
                pass
            return acc * 100, sen * 100, spec * 100, auc * 100, f1 * 100, bca * 100
        else:
            score = np.round(metrics(np.concatenate(y_true).reshape(-1, ).tolist(), np.concatenate(y_pred)).reshape(-1, ).tolist(), 5)[0]
        print('score:', score)
        # print('Accuracy of the network on the test subject  : {} %'.format(100 * correct / total))
        return score


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


def traintest_split_cross_subject(dataset, X, y, num_subjects, test_subject_id):
    X_c = X.copy()
    y_c = y.copy()
    test_x = X_c.pop(test_subject_id)
    test_y = y_c.pop(test_subject_id)
    train_x = np.concatenate(X_c, axis=0)
    train_y = np.concatenate(y_c, axis=0)

    print('Test subject s' + str(test_subject_id))
    print('Training/Test split:', train_x.shape, test_x.shape)
    return train_x, train_y, test_x, test_y


class EEGNet(nn.Module):

    def __init__(self,
                 n_classes: int,
                 Chans: int,
                 Samples: int,
                 kernLenght: int,
                 F1: int,
                 D: int,
                 F2: int,
                 dropoutRate:  float,
                 norm_rate: float):
        super(EEGNet, self).__init__()

        self.n_classes = n_classes
        self.Chans = Chans
        self.Samples = Samples
        self.kernLenght = kernLenght
        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.dropoutRate = dropoutRate
        self.norm_rate = norm_rate

        self.block1 = nn.Sequential(
            nn.ZeroPad2d((self.kernLenght // 2 - 1,
                          self.kernLenght - self.kernLenght // 2, 0,
                          0)),  # left, right, up, bottom
            nn.Conv2d(in_channels=1,
                      out_channels=self.F1,
                      kernel_size=(1, self.kernLenght),
                      stride=1,
                      bias=False),
            nn.BatchNorm2d(num_features=self.F1),
            # DepthwiseConv2d
            nn.Conv2d(in_channels=self.F1,
                      out_channels=self.F1 * self.D,
                      kernel_size=(self.Chans, 1),
                      groups=self.F1,
                      bias=False),
            nn.BatchNorm2d(num_features=self.F1 * self.D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(p=self.dropoutRate))

        self.block2 = nn.Sequential(
            nn.ZeroPad2d((7, 8, 0, 0)),
            # SeparableConv2d
            nn.Conv2d(in_channels=self.F1 * self.D,
                      out_channels=self.F1 * self.D,
                      kernel_size=(1, 16),
                      stride=1,
                      groups=self.F1 * self.D,
                      bias=False),
            nn.Conv2d(in_channels=self.F1 * self.D,
                      out_channels=self.F2,
                      kernel_size=(1, 1),
                      stride=1,
                      bias=False),
            nn.BatchNorm2d(num_features=self.F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(self.dropoutRate))

        self.classifier_block = nn.Sequential(
            nn.Linear(in_features=self.F2 * (self.Samples // (4 * 8)),
                    out_features=self.n_classes,
                    bias=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.block1(x)
        output = self.block2(output)
        output = output.reshape(output.size(0), -1)
        output = self.classifier_block(output)
        return output


class EEGWaveNet(nn.Module):
    def __init__(self, n_chans, n_classes):
        super(EEGWaveNet, self).__init__()

        self.temp_conv1 = nn.Conv1d(n_chans, n_chans, kernel_size=2, stride=2, groups=n_chans)
        self.temp_conv2 = nn.Conv1d(n_chans, n_chans, kernel_size=2, stride=2, groups=n_chans)
        self.temp_conv3 = nn.Conv1d(n_chans, n_chans, kernel_size=2, stride=2, groups=n_chans)
        self.temp_conv4 = nn.Conv1d(n_chans, n_chans, kernel_size=2, stride=2, groups=n_chans)
        self.temp_conv5 = nn.Conv1d(n_chans, n_chans, kernel_size=2, stride=2, groups=n_chans)
        self.temp_conv6 = nn.Conv1d(n_chans, n_chans, kernel_size=2, stride=2, groups=n_chans)

        self.chpool1 = nn.Sequential(
            nn.Conv1d(n_chans, 32, kernel_size=4, groups=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.01),
            nn.Conv1d(32, 32, kernel_size=4, groups=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.01))

        self.chpool2 = nn.Sequential(
            nn.Conv1d(n_chans, 32, kernel_size=4, groups=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.01),
            nn.Conv1d(32, 32, kernel_size=4, groups=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.01))

        self.chpool3 = nn.Sequential(
            nn.Conv1d(n_chans, 32, kernel_size=4, groups=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.01),
            nn.Conv1d(32, 32, kernel_size=4, groups=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.01))

        self.chpool4 = nn.Sequential(
            nn.Conv1d(n_chans, 32, kernel_size=4, groups=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.01),
            nn.Conv1d(32, 32, kernel_size=4, groups=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.01))

        self.chpool5 = nn.Sequential(
            nn.Conv1d(n_chans, 32, kernel_size=4, groups=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.01),
            nn.Conv1d(32, 32, kernel_size=4, groups=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.01))

        self.classifier = nn.Sequential(
            nn.Linear(160, 64),
            nn.LeakyReLU(0.01),
            nn.Linear(64, 32),
            nn.Sigmoid(),
            nn.Linear(32, n_classes))

    def forward(self, x, return_feature=False):
        x = x.squeeze(2)
        temp_x = self.temp_conv1(x)
        temp_w1 = self.temp_conv2(temp_x)
        temp_w2 = self.temp_conv3(temp_w1)
        temp_w3 = self.temp_conv4(temp_w2)
        temp_w4 = self.temp_conv5(temp_w3)
        temp_w5 = self.temp_conv6(temp_w4)

        w1 = self.chpool1(temp_w1).mean(dim=(-1))
        w2 = self.chpool2(temp_w2).mean(dim=(-1))
        w3 = self.chpool3(temp_w3).mean(dim=(-1))
        w4 = self.chpool4(temp_w4).mean(dim=(-1))
        w5 = self.chpool5(temp_w5).mean(dim=(-1))

        concat_vector = torch.cat([w1, w2, w3, w4, w5], 1)
        classes = self.classifier(concat_vector)

        return classes


def eeg_dnn(dataset, approach, cuda_device_id):
    if dataset == 'NICU' or dataset == 'CHSZ':
        X, y, sample_rate = process_seizure_data(dataset)
        paradigm = 'Seizure'
        num_subjects = len(X)
        ch_num = X[0].shape[1]
        eval_metrics = ['accs', 'sens', 'specs', 'aucs', 'f1s', 'bcas']

    scores_arr = []
    scores_arr_all = [[], [], [], [], [], []]

    for i in range(num_subjects):

        train_x, train_y, test_x, test_y = traintest_split_cross_subject(dataset, X, y, num_subjects, i)

        class_out = len(np.unique(train_y))

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

        if approach == 'EEGNet':
            model = EEGNet(n_classes=class_out,
                           Chans=train_x.shape[1],
                           Samples=train_x.shape[2],
                           kernLenght=int(sample_rate // 2),
                           F1=4,
                           D=2,
                           F2=8,
                           dropoutRate=0.25,
                           norm_rate=0.5)
        elif approach == 'EEGWaveNet':
            model = EEGWaveNet(n_chans=ch_num,
                               n_classes=class_out)

        seed_arr = np.arange(3)
        rand_init_scores = []
        accs, sens, specs, aucs, f1s, bcas = [], [], [], [], [], []

        for seed in seed_arr:
            if dataset == 'NICU':
                batch_size = 128
            elif dataset == 'CHSZ':
                batch_size = 32
            acc, sen, spec, auc, f1, bca = nn_fixepoch(model=model,
                                                       learning_rate=0.001,
                                                       num_iterations=10,
                                                       batch_size=batch_size,
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

        mind = 0
        for mets in [accs, sens, specs, aucs, f1s, bcas]:
            score = np.round(np.average(mets), 5)
            scores_arr_all[mind].append(mets)
            print(eval_metrics[mind], score)
            mind += 1

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

    dataset_arr = ['CHSZ', 'NICU']

    for dataset in dataset_arr:
        for approach in ['EEGWaveNet', 'EEGNet']:
            all_scores = []

            print(dataset, approach)

            eeg_dnn(dataset, approach, cuda_device_id)