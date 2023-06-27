# coding: utf-8
"""extract frequency domain features from 1*n signal.
"""

import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    arr1 = np.array([0., 2., 2.30769231, 3., 2., 2.5, 2.77777778, 2.66666667, 3.125, 2.14285714, 2.18181818, 2.41666667,
                      3., 2.57142857, 2.25, 2.85714286, 2.71428571, 3.2, 3.5, 2.875, 3.125, 3., 2.8, 1.88888889,
                      3.16666667, 2.17647059, 2.08333333, 3.375, 2., 1.72727273, 2.5, 2.57142857, 3.125, 2.625, 2.8,
                      2.14285714, 2.57142857, 2.30769231, 3., 3.42857143, 3.8, 2.66666667, 2.28571429, 3.42857143,
                      3.33333333, 2.8, 3.5, 1.5, 2.63636364, 2.5, 2.58823529, 2.5, 2.53846154, 2.35, 1.9047619, 2.65,
                      2.17647059, 2.40625, 2.43478261, 2.30434783, 2.27586207, 2.56097561, 2.77777778, 2.52777778,
                      2.38461538, 2.45945946, 2.31578947, 2.43396226, 2.58139535, 2.32608696, 2.86363636, 2.78409091,
                      1.57081545])  # pick an active series as example
    Fs = 1  # Hz
    N = arr1.shape[0]  # number of points to simulate, and our FFT size

    # add hamming window to avoid shift
    arr1 = arr1 * np.hamming(N)

    # FFT
    S = np.fft.fftshift(np.fft.fft(arr1))
    S_mag = np.abs(S)
    S_phase = np.angle(S)

    # plot magnitude and phase
    # f = np.arange(Fs / -2, Fs / 2, Fs / N)
    # plt.figure(0)
    # plt.plot(f, S_mag, '.-')
    # plt.xlabel('FFT index')
    # plt.ylabel('FFT of Signal (Magnitude)')
    # plt.figure(1)
    # plt.plot(f, S_phase, '.-')
    # plt.xlabel('FFT index')
    # plt.ylabel('FFT of Signal (Phase)')
    # plt.show()

    # features
    mean1 = np.mean(S_mag)
    max1 = np.max(S_mag)
    min1 = np.min(S_mag)
    var1 = np.var(S_mag)
    print(mean1, max1, min1, var1)
