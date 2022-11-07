import torch.nn as nn
import torch

class FC(nn.Module):
    def __init__(self, nn_in, nn_out):
        super(FC, self).__init__()
        self.fc = nn.Linear(nn_in, nn_out)

    def forward(self, x):
        x = self.fc(x)
        return x


class FC_ELU(nn.Module):
    def __init__(self, nn_in, nn_out):
        super(FC_ELU, self).__init__()
        # FC Layer
        self.fc = nn.Linear(nn_in, nn_out)
        self.actv = nn.ELU()

    def forward(self, x):
        x = self.fc(x)
        x = self.actv(x)
        return x


class FC_ELU_Dropout(nn.Module):
    def __init__(self, nn_in, nn_out):
        super(FC_ELU_Dropout, self).__init__()
        # FC Layer
        self.dropout = nn.Dropout(p=0.25, inplace=False)
        self.fc = nn.Linear(nn_in, nn_out)
        self.actv = nn.ELU()

    def forward(self, x):
        x = self.dropout(x)
        x = self.fc(x)
        x = self.actv(x)
        return x


class FC_middlecat(nn.Module):
    def __init__(self, nn_out, deep_feature_size, trad_feature_size):
        super(FC_middlecat, self).__init__()
        self.fc1 = nn.Linear(deep_feature_size, 32)
        self.fc2 = nn.Linear(trad_feature_size, deep_feature_size)
        #self.fc3 = nn.Linear(32 + trad_feature_size, nn_out)
        self.fc3 = nn.Linear(deep_feature_size + deep_feature_size, nn_out)
        self.actv = nn.ELU()

    def forward(self, x):
        x1, x2 = x
        #x1 = self.fc1(x1)
        #x1 = self.actv(x1)
        x2 = self.fc2(x2)
        x2 = self.actv(x2)
        catted = torch.cat((x1, x2), 1)
        x = self.fc3(catted)
        return x


class FC_2layer(nn.Module):
    def __init__(self, nn_in, nn_out):
        super(FC_2layer, self).__init__()
        self.fc1 = nn.Linear(nn_in, 16)
        self.fc2 = nn.Linear(16, nn_out)
        #self.actv = nn.ELU()

    def forward(self, x):
        x = self.fc1(x)
        #x = self.actv(x)
        x = self.fc2(x)
        return x