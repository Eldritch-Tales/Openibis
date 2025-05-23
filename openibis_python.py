import numpy as np
from scipy.signal import butter, filtfilt, blackman, fftconvolve
from scipy.fft import fft
from scipy.stats import trim_mean
from scipy.ndimage import uniform_filter1d
import math

# Main function: openibis
def openibis(eeg):
    Fs, stride = 128, 0.5
    BSRmap, BSR = suppression(eeg, Fs, stride)
    components = log_power_ratios(eeg, Fs, stride, BSRmap)
    depth_of_anesthesia = mixer(components, BSR)
    return depth_of_anesthesia

# Suppression function: Detects burst suppression and calculates BSR
def suppression(eeg, Fs, stride):
    N, n_stride = n_epochs(eeg, Fs, stride)
    BSRmap = np.zeros(N)
    for n in range(N):
        x = segment(eeg, n + 6.5, 2, n_stride)
        BSRmap[n] = np.all(np.abs(x - baseline(x)) <= 5)
    window = int(63 / stride)
    BSR = 100 * uniform_filter1d(BSRmap, size=window, origin=-(window - 1))
    return BSRmap, BSR

# Calculate the number of epochs
def n_epochs(eeg, Fs, stride):
    n_stride = int(Fs * stride)
    N = math.floor((len(eeg) - Fs) / n_stride) - 10
    return N, n_stride

# Extract a segment of the EEG data
def segment(eeg, start, number, n_stride):
    start_index = int(start * n_stride)
    end_index = start_index + int(number * n_stride)
    return eeg[start_index:end_index]

# Baseline correction for EEG
def baseline(x):
    v = np.vstack([np.arange(1, len(x) + 1)**p for p in range(2)]).T
    coeffs = np.linalg.lstsq(v, x, rcond=None)[0]
    return v @ coeffs