# Author: Sriram Schelbert
# Date: May 23, 2025


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

# Log-power ratios calculation
def log_power_ratios(eeg, Fs, stride, BSRmap):
    N, n_stride = n_epochs(eeg, Fs, stride)
    B, A = butter(2, 0.65 / (Fs / 2), btype='high')
    eeg_hi = filtfilt(B, A, eeg)

    psd = np.full((N, int(4 * n_stride / 2)), np.nan)
    suppression_filter = piecewise(np.arange(0, 64, 0.5), [0, 3, 6], [0, 0.25, 1]) ** 2
    components = np.full((N, 3), np.nan)

    for n in range(N):
        if is_not_burst_suppressed(BSRmap, n, 4):
            seg = segment(eeg_hi, n + 4, 4, n_stride)
            psd[n, :] = power_spectral_density(seg)

            if sawtooth_detector(segment(eeg, n + 4, 4, n_stride), n_stride):
                psd[n, :] *= suppression_filter

        thirty_sec = time_range(30, n, stride)
        try:
            vhigh_band = band_range(39.5, 46.5, 0.5)
            vhigh_band_alt = band_range(40, 47, 0.5)
            whole_band = band_range(0.5, 46.5, 0.5)
            whole_band_alt = band_range(1, 47, 0.5)
            mid_band = band_range(11, 20, 0.5)

            vhigh = np.sqrt(np.mean(psd[thirty_sec][:, vhigh_band] * psd[thirty_sec][:, vhigh_band_alt], axis=1))
            whole = np.sqrt(np.mean(psd[thirty_sec][:, whole_band] * psd[thirty_sec][:, whole_band_alt], axis=1))
            mid_power = prctmean(np.nanmean(10 * np.log10(psd[thirty_sec][:, mid_band]), axis=0), 50, 100)

            components[n, 0] = mean_band_power(psd[thirty_sec], 30, 47, 0.5) - mid_power
            components[n, 1] = trim_mean(10 * np.log10(vhigh / whole), 0.5)
            components[n, 2] = mean_band_power(psd[thirty_sec], 0.5, 4, 0.5) - mid_power
        except:
            pass  # Handle NaNs or range issues gracefully

    return components

# Calculate the Power Spectral Density (PSD)
def power_spectral_density(x):
    x = x - baseline(x)
    win = blackman(len(x))
    f = fft(win * x)
    return 2 * np.abs(f[:len(x)//2])**2 / (len(x) * np.sum(win**2))

# Sawtooth detection for K-complexes
def sawtooth_detector(eeg, n_stride):
    saw = np.concatenate([np.zeros(n_stride - 5), np.arange(1, 6)])
    saw = (saw - np.mean(saw)) / np.std(saw, ddof=1)
    r = np.arange(len(eeg) - len(saw))
    v = np.array([np.var(eeg[i:i+len(saw)]) for i in r])
    conv1 = fftconvolve(eeg, saw[::-1], mode='valid')
    conv2 = fftconvolve(eeg, saw, mode='valid')
    m = (np.vstack((conv1, conv2)) / len(saw))**2
    test = np.maximum((v > 10) * m[0] / v, (v > 10) * m[1] / v)
    return np.any(test > 0.63)

# Band range for frequency bands
def band_range(low, high, binsize):
    return np.arange(int(low / binsize), int(high / binsize) + 1)

# Mean power in a frequency band
def mean_band_power(psd, low, high, binsize):
    v = psd[:, band_range(low, high, binsize)]
    return np.nanmean(10 * np.log10(v[~np.isnan(v)]))

# Check if an epoch is burst-suppressed
def is_not_burst_suppressed(BSRmap, n, p):
    if n < p:
        return False
    return not np.any(BSRmap[n - p + 1:n + 1])

# Time range for a given number of seconds before the current epoch
def time_range(seconds, n, stride):
    return np.arange(max(0, int(n - seconds / stride) + 1), n + 1)

# Percentile-based mean calculation
def prctmean(x, lo, hi):
    x = np.array(x)
    lower = np.percentile(x, lo)
    upper = np.percentile(x, hi)
    return np.mean(x[(x >= lower) & (x <= upper)])

# Mixer: Calculates depth of anesthesia score
def mixer(components, BSR):
    sedation_score = scurve(components[:, 0], 104.4, 49.4, -13.9, 5.29)
    general_score = piecewise(components[:, 1], [-60.89, -30], [-40, 43.1])
    general_score += scurve(components[:, 1], 61.3, 72.6, -24.0, 3.55) * (components[:, 1] >= -30)

    bsr_score = piecewise(BSR, [0, 100], [50, 0])
    general_weight = piecewise(components[:, 2], [0, 5], [0.5, 1]) * (general_score < sedation_score)
    bsr_weight = piecewise(BSR, [10, 50], [0, 1])

    x = (sedation_score * (1 - general_weight)) + (general_score * general_weight)
    y = piecewise(x, [-40, 10, 97, 110], [0, 10, 97, 100]) * (1 - bsr_weight) + bsr_score * bsr_weight
    return y