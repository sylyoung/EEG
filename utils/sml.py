""" Implement the Spectral Meta Learner (SML) classifier.

SML infers base classifier performance from the empirical covariance matrix
and third central moment tensor of base classifier predictions.  Predictions of
base classifiers are assumed to be sample class labels, with 1 representing positive
class samples, and -1 negative class samples.  Moreover, SML
assumes that the base classifier predictions are conditionally independent.

THE RESULTS OF SML DEPEND ON THE VALIDITY OF THE CONDITIONAL INDEPENDENCE
ASSUMPTION.  IF YOUR DATA DOES NOT SATISFY THE CONDITIONAL INDEPENDENCE
ASSUMPTION, DO NOT USE SML.

In this implementation of SML [1], we estimate the rank one Eigenvalue and
Eigenvector from the empirical covariance matrix by the iterative strategy
proposed by Ahsen et al. [3].  The inferred Eigenvector elements are then
used as the aggregation weights for computing the SML score and for inferring
sample class labels.

We estimate the tensor singular value using a method introduced by Jaffe et al [2].
Using tensor singular value and the Eigenvalue from the covariance matrix
decomposition we estimate the balanced accuracy of each base classifier, and
the prevalence of postive class samples (class 1), without using labeled data.

References:
    1. Fabio Parisi, Francesco Strino, Boaz Nadler, and Yuval Kluger.
    Ranking and combining multiple predictors without labeled data.
    Proceedings of the National Academy of Sciences,
    111(4):1253--1258, 2014.

    2. Ariel Jaffe, Boaz Nadler, and Yuval Kluger.
    Estimating the accuracies of multiple classifiers without labeled data.
    Artificial Intelligence and Statistics, pages 407--415, 2015.

    3. Mehmet Eren Ahsen, Robert Vogel, and Gustavo Stolovitzky.
    Unsupervised evaluation and weighted aggregation of ranked predictions.
    arXiv preprint arXiv:1802.04684, 2018.

Available classes:
- sml: Implementation of the SML classifier
"""

import numpy as np


"""Test tensor inference and associated tools within 5 decimal places.

It has been shown that the covariance matrix of conditionally independent 
binary [1] and rank [2] base classifier predictions has a special structure.
This is that the off-diagonal elements i,j are proportional to the 
product of the i^th and j^th base classifier's performance as measured by
the balanced accuracy [1] or the difference of sample class conditioned average
rank predictions [2].  Consequently, these entries are those of a rank one 
matrix formed by the outer product of a vector encoding base classifier 
performances.

In this module we infer the performance vector from the empirical covariance
matrix by implementing the iterative procedue of Ahsen et al. [2].

References:
    1. Fabio Parisi, Francesco Strino, Boaz Nadler, and Yuval Kluger.
    Ranking and combining multiple predictors without labeled data.
    Proceedings of the National Academy of Sciences,
    111(4):1253--1258, 2014.

    2. Mehmet Eren Ahsen, Robert Vogel, and Gustavo Stolovitzky.
    Unsupervised evaluation and weighted aggregation of ranked predictions.
    arXiv preprint arXiv:1802.04684, 2018.

Available classes:
- Matrix
"""
import numpy as np


def infer_matrix(Q, tol, max_iter, return_iters=False):
    """Algorithm for inferring performance vector.

    Algorithm for inferring the diagonal entries which would make
    the covariance matrix Q, of full rank, a rank one matrix.

    Args:
        Q: The covariance matrix, ((M, M) ndarray)
        tol: The tolerance for convergence, convergence occurs
            when succesive eigenvalues are smaller than tol (float).
        max_iter: the maximum number of iterations of algorithm (integer)
        return_items: should the values of each iteration be returned
            (True or False, default False)

    Returns:
        Case 1, return_items = False:
            rank one matrix eigenvalue (float),
            eigenvector (ndarray)
            convergence message (str)
        Case 2, return_items = True:
            rank one matrix eigenvalue (float),
            eigenvector (ndarray),
            the inferred eigenvalue at each iteration,
            and a convergence message

    Raises:
        RuntimeError: raised when convergence criteria not met.
    """
    Q_off_diagonal = Q - np.diag(np.diag(Q))
    R = Q.copy()
    j = 0

    epsilon = np.sum(2 * np.diag(Q))
    eig_values = [epsilon]
    while epsilon > tol and j < max_iter:
        # decompose the rank one approximation
        R, eig_value, eig_vector = _update_r(Q_off_diagonal, R)
        epsilon = np.abs(eig_values[-1] - eig_value)
        eig_values += [eig_value]
        j += 1

    if j == max_iter:
        raise RuntimeError(("Matrix decomposition did not converge, try:\n"
                            "a) Increase the maximum number of"
                            " iterations above {:d}, or \n"
                            "b) increase the minimum"
                            " tolerance above {:.4f}.").format(max_iter, tol))

    # Assume that the majority of methods
    # correctly rank samples according to latent class
    # consequently the majority of Eigenvector
    # elements should be positive.
    if np.sum(eig_vector < 0) > Q.shape[0] / 2:
        eig_vector = -eig_vector

    if return_iters:
        return (eig_value, eig_vector, eig_values[1:],
                "Matrix decomposition converged in {} steps".format(j))
    else:
        return (eig_value, eig_vector,
                "Matrix decomposition converged in {} steps".format(j))


def _update_r(Q_off_diagonal, R):
    '''Update the estimate of the rank one matrix.

    Args:
        Q_off_diagonal: Covariance matrix with diagonal
            entries set to zero ((M, M) ndarray)
        R: estimated Rank one matrix ((M, M) ndarray)

    Returns:
        tuple with 3 entries
            0. Updated estimate of the rank one matrix R, ((M, M) ndarray)
            1. eigenvalue of R (float)
            2. eigenvector of R ((M,) ndarray)
    '''
    # spectral decomposition of a hermitian matrix
    l, v = np.linalg.eigh(R)

    # compute the diagonal of a rank one matrix
    rdiag = np.diag(l[-1] * v[:, -1] ** 2)

    # update the rank one matrix by replacing the diagonal
    # entries of the covariance matrix with those from the
    # previously estimated rank one matrix
    return (Q_off_diagonal + rdiag, l[-1], v[:, -1])


class Matrix:
    def __init__(self, max_iter=5000, tol=1e-6):
        '''Implement the iterative inference from Ahsen et al. [2].

        Args:
        max_iter: max number of iterations (int, default 5000)
        tol: stopping threshold (float, default 1e-6)

        Methods:
            fit: infer Eigenvector and Eigenvalue of rank one matrix
                corresponding to the empirical covariance matrix.
        '''
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, Q):
        """Find the diagonal entries that make Q a rank one matrix.

        Args:
            Q: The covariance matrix ((M, M) ndarray)
        """
        if Q.shape[0] < 3:
            raise ValueError("The minimum required number of base classifiers is 3.")
        if Q.ndim != 2:
            raise ValueError("Input ndarray must be a matrix (ndim == 2).")
        if Q.shape[0] != Q.shape[1]:
            raise ValueError("Covariance matrix is square, check input array.")
        self.eig_value, self.eig_vector, self.msg = infer_matrix(Q, self.tol, self.max_iter)


"""Find the singular value of the rank one third central moment tensor.

The third central moment of 3rd order conditionally independent base classifier
predictions has a special structure [1,2].  This is, the elements T_{ijk} for
i \neq j \neq k, are those of a rank one tensor,

T_{ijk} = l  v[i] v[j] v[k]

where l is the singular value and v is a unit norm vector.  In the case of SML and
SUMMA the unit norm vector consists fo values proportional to the performance
of each base classifier.  The peformance metric differs between problem statements.
In SML the base classifier predictions are binary (-1, 1), and the elements are proportional
to the balanced accuracy [1,2].  In SUMMA the base classifier predictions are sample rank
and the vector elements are proportional to Delta, the difference between the class
conditioned average rank values [3]. 

In this implementation we use the method from Jaffe et al. [1] to estimate the 
singular value (l) by linear regression.

References:
    1. Ariel Jaffe, Boaz Nadler, and Yuval Kluger.
    Estimating the accuracies of multiple classifiers without labeled data.
    Artificial Intelligence and Statistics, pages 407--415, 2015.

    2. Fabio Parisi, Francesco Strino, Boaz Nadler, and Yuval Kluger.
    Ranking and combining multiple predictors without labeled data.
    Proceedings of the National Academy of Sciences,
    111(4):1253--1258, 2014.

    3. Mehmet Eren Ahsen, Robert Vogel, and Gustavo Stolovitzky.
    Unsupervised evaluation and weighted aggregation of ranked predictions.
    arXiv preprint arXiv:1802.04684, 2018.

Available classes:
- Tensor
"""
from warnings import warn
import numpy as np
from scipy.special import comb


def check_inputs(tensor_shape, vector_shape):
    """Warnings and errors for data with 3 or less base classifiers."""
    if len(vector_shape) != 1:
        raise ValueError("Input vector should be an M length ndarray")
    M = vector_shape[0]
    if tensor_shape != (M, M, M):
        raise ValueError("Input tensor should be an MxMxM and"
                         "input vector M length ndarray")
    if M < 3:
        raise ValueError("The minimum required number of base classifiers is 3.")
    elif M == 3:
        warn("3 Base Classifiers may result in unreliable\n"
             "estimation of the positive class sample\n"
             "prevalence and the magnitude of each base\n"
             "classifiers performance.  The inferred weights used for\n"
             "computing the sample scores is unaffected.")


def get_tensor_idx(M):
    """Get indecies i \neq j \neq k of the 3rd order tensor (M, M, M) tensor T.

    Args:
        M: The number of entries of the tensor of interest (integer)

    Returns:
        idx : list of tuples containing indexes (i, j, k) such that i \neq j \neq k (list)
    """
    idx = list(range(comb(M, 3, exact=True)))

    l = 0
    for i in range(M - 2):
        for j in range(i + 1, M - 1):
            for k in range(j + 1, M):
                idx[l] = (i, j, k)
                l += 1
    return idx


class Tensor:
    """ Fit singular value of third central moment tensor.

    Fit the singular value (l) for the rank one tensor whose elements
    T_{i, j, k} = l * v[i] * v[j] * v[k] where i \neq j \neq k and v is
    the Eigenvector from the covariane decomposition.

    Args:
        T: third central moment tensor ((M, M, M) ndarray)
        v: Eigenvector of base classifier performances,
            from the matrix decomposition class ((M,) ndarray)

    Methods:
        tensor_and_eigenvector_elements: extract tensor values into array
        fit_singular_value: infer tensor singular by linear regression.

    Raises:
        ValueError: when the number of base classifiers
            is less than 3, tensor is not MxMxM, that
            T.shape = (M,M,M) and v.shape = (M,)
        UserWarning: when the number of base classifiers
            is equal to 3.
    """

    def __init__(self, T, v):
        check_inputs(T.shape, v.shape)
        self.tensorIndex = get_tensor_idx(T.shape[0])
        self.eigenvectorData, self.tensorData = self.tensor_and_eigenvector_elements(T, v)
        self.singular_value = self.fit_singular_value()

    def tensor_and_eigenvector_elements(self, T, v):
        """Extract tensor elements T_{ijk} for i \neq j \neq k.

        Args:
            T: third central moment tensor ((M, M , M) ndarray).
            v: Vector of unit norm. ((M,) ndarray).

        Returns:
            two element list [eigData, tData] as defined below (list)
                eigData : the product of vector entries v[i] * v[j] * v[k]
                    ((len(self.tensorIndex),) ndarray).
                tData : tensor elements i \neq j \neq k ((len(self.tensorIndex),) ndarray).
        """
        tData = np.zeros(len(self.tensorIndex))
        eigData = np.zeros(len(self.tensorIndex))

        # store tensor elements and the product of Eigenvector elements,
        # from each set of indices
        j = 0
        for widx in self.tensorIndex:
            tData[j] = T[widx[0], widx[1], widx[2]]
            eigData[j] = v[widx[0]] * v[widx[1]] * v[widx[2]]
            j += 1

        return (eigData, tData)

    def fit_singular_value(self):
        """Fit singular value by linear regression."""
        if len(self.eigenvectorData) == 1:
            return self.tensorData[0] / self.eigenvectorData[0]
        else:
            dy = np.sum(self.eigenvectorData * self.tensorData)
            dx = np.sum(self.eigenvectorData ** 2)
            return dy / dx


"""Check whether base classifier prediction data is correctly formatted.

Functions:
- _check_rank_data: Throws error if data are not an ndarray or sample ranks
- _check_binary_data: Throws error if data are not an ndarray or binary values [-1, 1].
"""

import numpy as np


def check_rank_data(data):
    """Test whether input data consists of rank values,
    that is specifically the integers 1,2,...,N.

    Args:
       data : (M method, N sample) ndarray of sample ranks
           with rows being base classifier rank predictions.

    Exceptions raised:
       TypeError : If data are not an numpy.ndarray
       ValueError : If data are not rank
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("Base classifier predictions must by numpy.ndarray")

    test_rank = np.arange(1, data.shape[1] + 1)  # Array to test user input data against
    for j in range(data.shape[0]):  # Test rank data for each method
        test_val = np.setdiff1d(test_rank, data[j, :]).size
        test_val += np.setdiff1d(data[j, :], test_rank).size
        if test_val > 0:
            raise ValueError("Base classifier predictions must be a ranked list")


def check_binary_data(data):
    """Test whether input data exclusively consists
    of values -1 or 1.

    Args:
        data : (M method, N sample) ndarray of
            binary predictions [-1, 1]

    Exceptions raised:
       TypeError : If data are not an numpy.ndarray
       ValueError : If data are not binary
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("Base classifier predictions must by numpy.ndarray")

    test_binary = np.array([-1, 1])  # Array to test user input data against
    for j in range(data.shape[0]):  # Test binary data for each method
        test_val = np.setdiff1d(test_binary, data[j, :]).size
        test_val += np.setdiff1d(data[j, :], test_binary).size
        if test_val > 0:
            raise ValueError("Base classifier predictions must be a binary values -1 or 1")


import numpy as np

def third(data):
    '''
    Input
    -----
    numpy array M base classifier x N samples numpy array

    Returns
    3rd order tensor of the third central moment
    '''
    M = data.shape[0]
    N = data.shape[1]
    # subtract mean from data
    tmp = data - np.tile(np.mean(data, 1).reshape(M, 1), (1, N))
    # instantiate tensor
    T = np.zeros(shape=(M, M, M))
    # loop over methods along the third mode
    for w in range(data.shape[0]):
        T[:,:, w] = np.dot(np.tile(tmp[w, :], (M, 1)) * tmp, tmp.T) / N
    return T


class Sml:
    """Apply SML ensemble to data.

    Args:
        prevalence: the fraction of samples from the  positive class.
            When None given, infer prevalence from data.
            (float beteen 0 and 1 or None, default None)

    Attributes:
    - method: name of classifier (str).
    - prevalence: fraction of samples from the positive class (float between 0 and 1)
    - metric: name of performance metric associated with inference (str)
    - cov: covariance decomposition class instance
    - tensor: tensor decomposition class instance

    Public Methods:
    - fit: Fit the SML model to the empirical covariance and third central moment.
    - get_ba: Compute and return the inferred balanced accuracy of each base
        classifier (ndarray).
    - get_prevalence: Compute and return inferred or return given sample class
        prevalence (float).
    - get_weights: Get the weights used for the SML weighted sum (ndarray).
    - get_scores: Compute and return the SML sample scores (ndarray).
    - get_inference: Compute and return the SML inferred sample class labels (ndarray).
    """

    def __init__(self, prevalence=None):
        self.method = 'SML'
        self.prevalence = prevalence
        self.metric = "BA"

    def fit(self, data, tol=1e-3, max_iter=500):
        """Fit the SML model to the empirical covariance and third central moment.

        Args:
            data : N sample predictions by M base classifiers ((M, N) ndarray)
            tol: the tolerance for matrix decomposition (float, default 1e-3).
            max_iter: of the maximum number of iterations for matrix decomposition
                (int, default 500).

        Raises:
            ValueError: when M methods < 5 or N samples < 5.
            ValueError: when data are not binary values [-1, 1].
        """
        check_binary_data(data)

        # Covariance decomposition
        self.cov = Matrix(tol=tol, max_iter=max_iter)
        self.cov.fit(np.cov(data))

        # tensor decomposition
        if self.prevalence is None:
            self.tensor = Tensor(third(data), self.cov.eig_vector)

    def get_ba(self):
        """Compute the balanced accuracy of each base classifier.

        Using the fitted Eigenvector and the norm of the
        performance vector compute the inferred balanced accuracy of
        each base classifier.

        Returns:
            (M,) ndarray of inferred balanced accuracies.
        """
        return 0.5 * (1 + self.cov.eig_vector * self.get_ba_norm())

    def get_ba_norm(self):
        '''Compute the norm of the performance vector.

        Case 1: known prevalence
            Return the norm of the performance vector using the a priori
            specified positive class prevalence and Eigenvalue from the
            matrix decomposition.
        Case 2: tensor singular value
            Return the inferred norm of the performance vector using the
            Eigenvalue and singular value estimated from the matrix and tensor
            decomposition, respectively.

        Returns:
            float, norm of the performance vector.
        '''
        if self.prevalence is not None:
            return np.sqrt(self.cov.eig_value / (4 * self.prevalence * (1 - self.prevalence)))
        else:
            beta = (self.tensor.singular_value / self.cov.eig_value) ** 2
            return 0.5 * np.sqrt(beta + 4 * self.cov.eig_value)

    def get_prevalence(self):
        """Return sample class prevalence.

        Case 1: known prevalence
            Return the a priori known positive class sample prevalence.
        Case 2: tensor singular value
            Return the inferred positive class sample prevalence.

        Returns:
            float, prevalence of the positive class on the interval [0, 1]
        """
        if self.prevalence is not None:
            return self.prevalence
        else:
            beta = self.tensor.singular_value / self.cov.eig_value
            return 0.5 * (1 - 0.5 * beta / self.get_ba_norm())

    def get_weights(self):
        '''Return the SML weights.

        Returns:
            (M,) ndarray SML weights
        '''
        return self.cov.eig_vector

    def get_scores(self, data):
        """ Compute SML score for each sample.

        The SML score is the value of approximate likelihood used
        to infer sample class labels in Parisi et al. [1].  Simply, the
        k^th samples score is the weighted sum of all base classifier
        predictions.

        Args:
            data: (M, N) ndarray of N binary value predictions for
                each M base classifier

        Returns:
            s: (N,) ndarray of SML scores for each sample

        Raises:
            ValueError: if data are not binary predictions
            TypeError: if data are not in ndarray
        """
        check_binary_data(data)

        M = data.shape[0]

        s = 0
        for j in range(M):
            s += self.cov.eig_vector[j] * data[j, :]
        return s

    def get_inference(self, data):
        """Compute and return the SML inferred sample class labels.

        The approximate maximum likelihood esimtate of sample class labels
        from Parisi et al. [1].  We assign samples with SML scores greater
        than or equal to zero with a positive class label (1), otherwise a
        negative class label (-1).

        Args:
            data : (M, N) ndarray of N binary value predictions of M base classifiers

        Returns:
            labels : (N,) ndarray of SML inferred binary values for each sample,
                1 designating positive class and -1 negative class.

        Raises:
            ValueError: if data are not binary predictions
            TypeError: if data are not in ndarray
        """
        labels = self.get_scores(data)
        labels[labels >= 0] = 1.
        labels[labels < 0] = -1.
        return labels