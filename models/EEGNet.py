import torch
import torch.nn as nn
import sys

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


class EEGNet_feature(nn.Module):

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
        super(EEGNet_feature, self).__init__()

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


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.block1(x)
        output = self.block2(output)
        output = output.reshape(output.size(0), -1)

        return output


class EEGNetSiameseFusion(nn.Module):

    def __init__(self,
                 n_classes,
                 Chans,
                 Samples,
                 kernLenght,
                 F1,
                 D: int,
                 F2: int,
                 dropoutRate: float,
                 norm_rate,
                 ch_num=None,
                 return_representation=False):
        super(EEGNetSiameseFusion, self).__init__()

        self.n_classes = n_classes
        self.Chans = Chans
        if ch_num:
            self.ch_num = ch_num
        else:
            self.ch_num = self.Chans
        self.Samples = Samples
        self.kernLenght = kernLenght
        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.norm_rate = norm_rate
        self.dropoutRate = dropoutRate

        # Knowledge Blocks
        self.kblock1 = nn.Sequential(
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
                      kernel_size=(self.ch_num, 1),
                      groups=self.F1,
                      bias=False),
            nn.BatchNorm2d(num_features=self.F1 * self.D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(p=self.dropoutRate))

        self.kblock2 = nn.Sequential(
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
            nn.Dropout(p=self.dropoutRate))

        # Data Blocks
        self.dblock1 = nn.Sequential(
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

        self.dblock2 = nn.Sequential(
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
            nn.Dropout(p=self.dropoutRate))

        self.kdrop1 = nn.Dropout(p=1.0)
        self.knodrop1 = nn.Dropout(p=0.0)

        self.kdrop2 = nn.Dropout(p=0.0)
        self.knodrop2 = nn.Dropout(p=1.0)

        self.ddrop1 = nn.Dropout(p=1.0)
        self.dnodrop1 = nn.Dropout(p=0.0)

        self.ddrop2 = nn.Dropout(p=0.0)
        self.dnodrop2 = nn.Dropout(p=1.0)

        self.classifier_block = nn.Sequential(
            nn.Linear(in_features=self.F2 * (self.Samples // (4 * 8)) * 2,
                    out_features=self.n_classes,
                    bias=True))

    def forward(self, x):
        knowledge, data = x
        assert len(knowledge) == len(data)

        knowledge = self.kblock1(knowledge)
        data = self.dblock1(data)

        assert len(knowledge) == len(data)

        """
        out_size = len(knowledge)
        bool_tensor = torch.cat((torch.zeros(out_size // 2, dtype=torch.bool), torch.ones(out_size // 2, dtype=torch.bool)))
        randomized_bool = torch.randperm(len(bool_tensor))
        indices = bool_tensor[randomized_bool]
        rest_indices = (torch.ones(out_size, dtype=torch.int) - indices.to(torch.int)).to(torch.bool)

        knowledgedrop = self.kdrop1(knowledge[indices])
        if self.training:
            knowledgedrop *= (1 / (1 - 0.5))
        knowledgenodrop = self.knodrop1(knowledge[rest_indices])
        if self.training:
            knowledgenodrop *= (1 / (1 - 0.5))

        datadrop = self.ddrop1(data[indices])
        if self.training:
            datadrop *= (1 / (1 - 0.5))
        datanodrop = self.dnodrop1(data[rest_indices])
        if self.training:
            datanodrop *= (1 / (1 - 0.5))

        out = []
        cnt1 = 0
        cnt2 = out_size // 2
        # TODO replace loop with matrix operation
        for i in range(out_size):
            if indices[i] == True:
                out.append(cnt1)
                cnt1 += 1
            else:
                out.append(cnt2)
                cnt2 += 1

        knowledge = torch.cat((knowledgedrop, knowledgenodrop))[out]
        data = torch.cat((datadrop, datanodrop))[out]
        """

        knowledge = self.kblock2(knowledge)
        data = self.dblock2(data)

        assert len(knowledge) == len(data)
        """
        out_size = len(knowledge)
        bool_tensor = torch.cat((torch.zeros(out_size // 2, dtype=torch.bool), torch.ones(out_size // 2, dtype=torch.bool)))
        randomized_bool = torch.randperm(len(bool_tensor))
        indices = bool_tensor[randomized_bool]
        rest_indices = (torch.ones(out_size, dtype=torch.int) - indices.to(torch.int)).to(torch.bool)

        knowledgedrop = self.kdrop2(knowledge[indices])
        if self.training:
            knowledgedrop *= (1 / (1 - 0.5))
        knowledgenodrop = self.knodrop2(knowledge[rest_indices])
        if self.training:
            knowledgenodrop *= (1 / (1 - 0.5))

        datadrop = self.ddrop2(data[indices])
        if self.training:
            datadrop *= (1 / (1 - 0.5))
        datanodrop = self.dnodrop2(data[rest_indices])
        if self.training:
            datanodrop *= (1 / (1 - 0.5))

        out = []
        cnt1 = 0
        cnt2 = out_size // 2
        # TODO replace loop with matrix operation
        for i in range(out_size):
            if indices[i] == True:
                out.append(cnt1)
                cnt1 += 1
            else:
                out.append(cnt2)
                cnt2 += 1

        knowledge = torch.cat((knowledgedrop, knowledgenodrop))[out]
        data = torch.cat((datadrop, datanodrop))[out]
        """

        knowledge = knowledge.reshape(knowledge.size(0), -1)
        data = data.reshape(data.size(0), -1)

        output = torch.cat((knowledge, data), 1)

        output = self.classifier_block(output)

        return output



class EEGNetCNNFusion(nn.Module):

    def __init__(self,
                 n_classes: int,
                 Chans: int,
                 Samples: int,
                 kernLenght: int,
                 F1: int,
                 D: int,
                 F2: int,
                 dropoutRate:  float,
                 norm_rate: float,
                 deep_dim: int):
        super(EEGNetCNNFusion, self).__init__()

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

        self.conv1 = nn.Conv1d(in_channels=self.Chans, out_channels=1, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.bn1 = nn.BatchNorm1d(num_features=74)
        self.fc = nn.Linear(in_features=74 + deep_dim,
                    out_features=self.n_classes,
                    bias=True)

    def forward(self,  x):
        feature, data = x

        feature = self.conv1(feature)
        feature = torch.flatten(feature, 1)
        feature = self.bn1(feature)

        data = self.block1(data)
        data = self.block2(data)
        data = data.reshape(data.size(0), -1)

        feature = feature.reshape(feature.size(0), -1)
        data = data.reshape(data.size(0), -1)

        output = torch.cat((feature, data), 1)

        output = self.fc(output)
        return output
