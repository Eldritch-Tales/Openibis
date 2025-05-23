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