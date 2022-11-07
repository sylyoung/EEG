import os
import numpy as np
import math
import scipy.io as sio
from scipy.fftpack import fft, ifft
import scipy

def de_feature_extractor(EEGdata, fs, fStart, fEnd, window, stftn):
    #assert EEGdata.shape[0] == 62

    '''
    fs = 200
    channels = 62
    fStart = [1, 4, 8, 14, 31]
    fEnd = [4, 8, 14, 31, 50]
    window = 1  # 窗口长度
    stftn = 512  # 频域采样率
    '''

    WindowPoints = fs * window  # 每个窗口采样点数

    fStartNum = np.zeros([1, len(fStart)])
    fEndNum = np.zeros([1, len(fEnd)])
    for i in range(len(fStart)):  # 频段划分
        fStartNum[0, i] = int(fStart[i] / fs * stftn)
        fEndNum[0, i] = int(fEnd[i] / fs * stftn) - 1

    n, m = EEGdata.shape
    l = int(m / WindowPoints)  # 汉宁窗个数
    psd = np.zeros([n, l, len(fStart)])

    # Hanning window
    Hlength = window * fs
    # Hwindow = hann(Hlength)  # 加窗

    WindowPoints = fs * window

    for j in range(l):  # 窗口
        # print("timepoints: %d s" %(j+1))
        #dataNow = EEGdata[:, WindowPoints * j:WindowPoints * (j + 1)]
        dataNow = EEGdata
        for k in range(n):  # 通道
            temp = dataNow[k, :]
            # Hdata = temp.*Hwindow.T
            Hdata = temp
            FFTdata = scipy.fftpack.fft(Hdata)
            freqs = scipy.fftpack.fftfreq(WindowPoints) * stftn

            # FFTdata = FFTdata.reshape(1, len(FFTdata))
            magFFTdata = np.abs(FFTdata[0:int(stftn / 2)])
            # magFFTdata = np.abs(FFTdata)
            for s in range(len(fStart)):  # 频段
                a = fStartNum[0, s]
                b = fEndNum[0, s]
                # print(a, b)
                c = magFFTdata[s]
                # print(c)
                E = 0
                E_log = 0
                for t in range(int(fStartNum[0, s]) - 1, int(fEndNum[0, s])):
                    # print(t+1)# 频率点
                    E = E + (magFFTdata[t] * magFFTdata[t])
                E = E / (fEndNum[0, s] - fStartNum[0, s] + 1)
                psd[k, j, s] = 10 * math.log(E+1e-9, 10)
                # print(psd[k,j,s])

            # print("j = %d,  k = %d" %(j+1, k+1))
            # print('fft')

    DEFeature = psd
    DEFeature = np.vstack((DEFeature[:, :, 0], DEFeature[:, :, 1], DEFeature[:, :, 2], DEFeature[:, :, 3], DEFeature[:, :, 4]))

    return DEFeature.T  # [dimensions, samples]