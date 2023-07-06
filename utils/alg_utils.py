import numpy as np
import torch

import torch.nn.functional as F
from scipy.linalg import fractional_matrix_power


def log_euclid_mean(X):
    """
    Log-Euclidean mean https://github.com/alexandrebarachant/covariancetoolbox/blob/master/lib/mean/logeuclid_mean.m
    Parameters
    ----------
    X : numpy array
        data of shape (num_samples, num_channels, num_channels)

    Returns
    ----------
    cov : numpy array
        data of shape (num_channels, num_channels)
    """

    K = X.shape[0]
    fc = np.zeros((X.shape[1], X.shape[1]))
    for i in range(K):
        positive_array = np.clip(X[i], 1, None)  # TODO
        logvalue = np.log(positive_array)
        fc += logvalue
    cov = np.exp(fc / K)
    return cov


def LA(Xs, Ys, Xt, Yt):
    """
    Label Alignment
    Parameters
    ----------
    Xs : numpy array
        data of shape (num_samples, num_channels, num_time_samples)
    Xt : numpy array
        data of shape (num_samples, num_channels, num_time_samples)
    Ys : numpy array, label of 0, 1, ... (int)
        data of shape (num_samples, )
    Yt : numpy array, label of 0, 1, ... (int)
        data of shape (num_samples, )

    Returns
    ----------
    XLA : numpy array
        data of shape (num_samples, num_channels, num_time_samples)
    YLA : numpy array
        data of shape (num_samples, )
    """

    assert Xs.shape[1] == Xt.shape[1], print('LA Error, channel mismatch!')
    assert Xs.shape[2] == Xt.shape[2], print('LA Error, time sample mismatch!')
    label_space_s, cnts_class_s = np.unique(Ys, return_counts=True)
    label_space_t, cnts_class_t = np.unique(Yt, return_counts=True)
    assert len(label_space_s) == len(label_space_t), print('LA Error, label space mismatch!')
    num_classes = len(label_space_s)

    Xs_by_labels = []
    Xt_by_labels = []
    for c in range(num_classes):
        inds_class = np.where(Ys == c)[0]
        Xs_by_labels.append(Xs[inds_class])
        inds_class = np.where(Yt == c)[0]
        Xt_by_labels.append(Xt[inds_class])

    covxs = []
    covxt = []

    for c in range(num_classes):
        covxs.append(np.zeros((cnts_class_s[c], Xs.shape[1], Xs.shape[1])))
        covxt.append(np.zeros((cnts_class_t[c], Xt.shape[1], Xt.shape[1])))

    XLA = []
    YLA = []

    for c in range(num_classes):
        for i in range(len(covxs[c])):
            covxs[c][i] = np.cov(Xs_by_labels[c][i])
        for i in range(len(covxt[c])):
            covxt[c][i] = np.cov(Xt_by_labels[c][i])

        #covxs_class = log_euclid_mean(covxs[c]) # TODO log-euclid mean negative values due to non-symmetric positive definite cov matrix
        covxs_class = np.mean(covxs[c], axis=0)
        sqrtCs = fractional_matrix_power(covxs_class, -0.5)
        #covxt_class = log_euclid_mean(covxt[c])
        covxt_class = np.mean(covxt[c], axis=0)
        sqrtCt = fractional_matrix_power(covxt_class, 0.5)
        A = np.dot(sqrtCt, sqrtCs)
        for i in range(len(Xs_by_labels[c])):
            XLA.append(np.dot(A, Xs_by_labels[c][i]))
            YLA.append(c)

    XLA = np.array(XLA)
    YLA = np.array(YLA).reshape(-1,)
    assert XLA.shape == Xs.shape, print('LA Error, X shape problem!')
    assert YLA.shape == Ys.shape, print('LA Error, Y shape problem!')
    assert np.unique(YLA, return_counts=True)[1][0] == cnts_class_s[0], print('LA Error, labels problem!')

    return XLA, YLA


def EA(x):
    """
    Parameters
    ----------
    x : numpy array
        data of shape (num_samples, num_channels, num_time_samples)

    Returns
    ----------
    XEA : numpy array
        data of shape (num_samples, num_channels, num_time_samples)
    """
    cov = np.zeros((x.shape[0], x.shape[1], x.shape[1]))
    for i in range(x.shape[0]):
        cov[i] = np.cov(x[i])
    refEA = np.mean(cov, 0)
    sqrtRefEA = fractional_matrix_power(refEA, -0.5)
    XEA = np.zeros(x.shape)
    for i in range(x.shape[0]):
        XEA[i] = np.dot(sqrtRefEA, x[i])
    return XEA


def EA_online(x, R, sample_num):
    """
    Parameters
    ----------
    x : numpy array
        sample of shape (num_channels, num_time_samples)
    R : numpy array
        current reference matrix (num_channels, num_channels)
    sample_num: int
        previous number of samples used to calculate R

    Returns
    ----------
    refEA : numpy array
        data of shape (num_channels, num_channels)
    """

    cov = np.cov(x)
    refEA = (R * sample_num + cov) / (sample_num + 1)
    return refEA


def soft_cross_entropy_loss(input, target):
    logprobs = torch.nn.functional.log_softmax(input, dim=1)
    return -(target * logprobs).sum() / input.shape[0]


def cross_entropy_with_probs(
        input,
        target,
        weight=None,
        reduction="mean"):
    """Calculate cross-entropy loss when targets are probabilities (floats), not ints.
    PyTorch's F.cross_entropy() method requires integer labels; it does accept
    probabilistic labels. We can, however, simulate such functionality with a for loop,
    calculating the loss contributed by each class and accumulating the results.
    Libraries such as keras do not require this workaround, as methods like
    "categorical_crossentropy" accept float labels natively.
    Note that the method signature is intentionally very similar to F.cross_entropy()
    so that it can be used as a drop-in replacement when target labels are changed from
    from a 1D tensor of ints to a 2D tensor of probabilities.
    Parameters
    ----------
    input
        A [num_points, num_classes] tensor of logits
    target
        A [num_points, num_classes] tensor of probabilistic target labels
    weight
        An optional [num_classes] array of weights to multiply the loss by per class
    reduction
        One of "none", "mean", "sum", indicating whether to return one loss per data
        point, the mean loss, or the sum of losses
    Returns
    -------
    torch.Tensor
        The calculated loss
    Raises
    ------
    ValueError
        If an invalid reduction keyword is submitted
    """

    num_points, num_classes = input.shape
    # Note that t.new_zeros, t.new_full put tensor on same device as t

    cum_losses = input.new_zeros(num_points)
    for y in range(num_classes):
        target_temp = input.new_full((num_points,), y, dtype=torch.long)
        y_loss = F.cross_entropy(input, target_temp, reduction="none")
        if weight is not None:
            y_loss = y_loss * weight[y]
        cum_losses += target[:, y].float() * y_loss

    if reduction == "none":
        return cum_losses
    elif reduction == "mean":
        return cum_losses.mean()
    elif reduction == "sum":
        return cum_losses.sum()
    else:
        raise ValueError("Keyword 'reduction' must be one of ['none', 'mean', 'sum']")


def calc_distance_wave(data):
    """
    :param data: np array, 1-d array representing a wave
    :return: int, distance of walking on the wave from start point to the end
    """
    total_dist = 0
    for i in range(len(data) - 1):
        dist = np.sqrt(1 + np.square(data[i + 1] - data[i]))
        total_dist += dist
    return total_dist



