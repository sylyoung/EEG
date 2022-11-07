import torch


class RBM():
    def __init__(self, nv, nh):
        """
        Parameters
        ----------
        nv : int
            number of visible nodes
        nh : int
            number of hidden nodes
        """

        self.W = torch.randn(nh, nv)
        # the probability of the hidden nodes given the visible nodes
        self.a = torch.randn(1, nh)
        # the probability of the visible nodes given the hidden nodes
        self.b = torch.randn(1, nv)


    def sample_h(self, x):
        """
        Parameters
        ----------
        x : numpy array
            visible neurons

        Returns
        ----------
        p_h_given_v : probability of h given v where h and v represent the hidden and visible nodes respectively
        torch.bernoulli(p_h_given_v) : bernoulli samples of the hidden neurons, for binary classification
        """

        wx = torch.mm(x, self.W.t())
        activation = wx + self.a.expand_as(wx)
        p_h_given_v = torch.sigmoid(activation)
        return p_h_given_v, torch.bernoulli(p_h_given_v)

    def sample_v(self, y):
        """
        Parameters
        ----------
        y : numpy array
            visible neurons

        Returns
        ----------
        p_v_given_h : probability of v given h where h and v represent the hidden and visible nodes respectively
        torch.bernoulli(p_v_given_h) : bernoulli samples of the visible neurons, for binary classification
        """

        wy = torch.mm(y, self.W)
        activation = wy + self.b.expand_as(wy)
        p_v_given_h = torch.sigmoid(activation)
        return p_v_given_h, torch.bernoulli(p_v_given_h)

    def train(self, v0, vk, ph0, phk):
        """
        Parameters
        ----------
        y : numpy array
            the input vector
        vk : numpy array
            the visible nodes obtained after k samplings
        ph0 : numpy array
            the vector of probabilities
        phk : numpy array
            the probabilities of the hidden nodes after k samplings.
        """

        self.W += torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)
        self.b += torch.sum((v0 - vk), 0)
        self.a += torch.sum((ph0 - phk), 0)