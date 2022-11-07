
import moabb
import numpy as np
import mne

from moabb.datasets import BNCI2014001, BNCI2014002, BNCI2014008, BNCI2014009, BNCI2015003, BNCI2015004, EPFLP300
from moabb.paradigms import MotorImagery, P300


def dataset_to_file(dataset_name):
    moabb.set_log_level("ERROR")

    if dataset_name == 'BNCI2014001':
        dataset = BNCI2014001()
        paradigm = MotorImagery(n_classes=4)
        # (5184, 22, 1001) (5184,) 250Hz 9subjects * 4classes * (72+72)trials for 2sessions
    elif dataset_name == 'BNCI2014002':
        dataset = BNCI2014002()
        paradigm = MotorImagery(n_classes=2)
        # (2240, 15, 2561) (2240,) 512Hz 14subjects * 2classes * (50+30)trials * 2sessions(not namely separately)

    X, labels, meta = paradigm.get_data(dataset=dataset, subjects=dataset.subject_list[:])
    ar_unique, cnts = np.unique(labels, return_counts=True)
    print("labels:", ar_unique)
    print("Counts:", cnts)
    print(X.shape, labels.shape)
    np.save('./' + dataset_name + '_X', X)
    np.save('./' + dataset_name + '_labels', labels)
    meta.to_csv('./' + dataset_name + '_meta.csv')

    if isinstance(paradigm, MotorImagery):
        X, labels, meta = paradigm.get_data(dataset=dataset, subjects=dataset.subject_list[:], return_epochs=True)
        return X.info

def data_loader(dataset):

    mne.set_log_level('warning')

    X = np.load('./data/' + dataset + '/X.npy')
    y = np.load('./data/' + dataset + '/labels.npy')
    print(X.shape, y.shape)

    if dataset == 'BNCI2014001':
        paradigm = 'MI'
        num_subjects = 9

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

    return X, y, num_subjects


if __name__ == '__main__':
    dataset_name = 'BNCI2014001'
    #dataset_name = 'BNCI2014002'

    info = dataset_to_file(dataset_name)
    print(info)

    X, y, num_subjects = data_loader(dataset_name)
    print('data/labels/num_subjects:', X.shape, y.shape, num_subjects)