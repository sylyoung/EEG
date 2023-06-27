import torch
import torch.nn as nn
import sys
from EEG.models.BayesianCNN.layers.BBB.BBBConv import BBBConv2d
from EEG.models.BayesianCNN.layers.BBB.BBBLinear import BBBLinear
from EEG.models.BayesianCNN.layers.misc import FlattenLayer, ModuleWrapper


class BEEGNet(ModuleWrapper):

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
        super().__init__()

        self.n_classes = n_classes
        self.Chans = Chans
        self.Samples = Samples
        self.kernLenght = kernLenght
        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.dropoutRate = dropoutRate
        self.norm_rate = norm_rate

        self.conv = nn.Sequential(
            nn.ZeroPad2d((self.kernLenght // 2 - 1,
                          self.kernLenght - self.kernLenght // 2, 0,
                          0)),  # left, right, up, bottom
            BBBConv2d(in_channels=1,
                      out_channels=self.F1,
                      kernel_size=(1, self.kernLenght),
                      stride=1,
                      bias=False),
            nn.BatchNorm2d(num_features=self.F1),
            # DepthwiseConv2d
            BBBConv2d(in_channels=self.F1,
                      out_channels=self.F1 * self.D,
                      kernel_size=(self.Chans, 1),
                      groups=self.F1,
                      bias=False),
            nn.BatchNorm2d(num_features=self.F1 * self.D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(p=self.dropoutRate),

            nn.ZeroPad2d((7, 8, 0, 0)),
            # SeparableConv2d
            BBBConv2d(in_channels=self.F1 * self.D,
                      out_channels=self.F1 * self.D,
                      kernel_size=(1, 16),
                      stride=1,
                      groups=self.F1 * self.D,
                      bias=False),
            BBBConv2d(in_channels=self.F1 * self.D,
                      out_channels=self.F2,
                      kernel_size=(1, 1),
                      stride=1,
                      bias=False),
            nn.BatchNorm2d(num_features=self.F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(self.dropoutRate),

            FlattenLayer(self.F2 * (self.Samples // (4 * 8))),
            BBBLinear(in_features=self.F2 * (self.Samples // (4 * 8)),
                      out_features=self.n_classes,
                      bias=True)

        )

