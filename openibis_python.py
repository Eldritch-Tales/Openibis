# Author: Sriram Schelbert
# Date: May 23, 2025

import numpy as np
from scipy.signal import butter, filtfilt, fftconvolve, lfilter
from scipy.fft import fft
from scipy.stats import trim_mean
from scipy.ndimage import uniform_filter1d
# from scipy.io import loadmat
import math
import matplotlib.pyplot as plt

# Main function: openibis
def openibis(eeg_input):

    eeg = np.asarray(eeg_input).squeeze()

    Fs, stride = 128, 0.5
    BSRmap, BSR = suppression(eeg, Fs, stride)

    print("BSRmap[:100]:", BSRmap[:100])
    print("Suppressed epochs:", np.sum(BSRmap))
    print("Total epochs:", len(BSRmap))

    time = np.arange(len(eeg)) * stride

    components = log_power_ratios(eeg, Fs, stride, BSRmap)
    # components = np.nan_to_num(components, nan=0.0)

    depth_of_anesthesia = mixer(components, BSR)

    # Plot the components against each other to diagnose issues
    plt.figure(figsize=(10, 6))
    plt.plot(components[:, 0], label='Component 0: sedation')
    plt.plot(components[:, 1], label='Component 1: general')
    plt.plot(components[:, 2], label='Component 2: bsr weight')
    plt.title("DOA Components over Time")
    plt.legend()
    plt.show()

    return depth_of_anesthesia

def suppression(eeg, Fs, stride):
    N, n_stride = n_epochs(eeg, Fs, stride)
    BSRmap = np.zeros(N)
    for n in range(N):
        x = segment(eeg, n + 6.5, 2, n_stride)
        BSRmap[n] = np.all(np.abs(x - baseline(x)) <= 5)
    window = int(round(63 / stride))
    BSR = 100 * causal_moving_average(BSRmap, window)
    return BSRmap, BSR

def causal_moving_average(x, window_size):
    # Pad the left side with zeros to maintain alignment (like MATLAB's movmean(..., [M 0]))
    padded = np.pad(x, (window_size - 1, 0), mode='constant', constant_values=0)
    smoothed = uniform_filter1d(padded, size=window_size, origin=0)[(window_size - 1):]
    return smoothed

# Calculate the number of epochs
def n_epochs(eeg, Fs, stride):
    n_stride = int(round(Fs * stride))
    N = math.floor((len(eeg) - Fs) / n_stride) - 10
    return N, n_stride

# Extract a segment of the EEG data
def segment(eeg, start, number, n_stride):
    start_index = int(round(start * n_stride))
    end_index = start_index + int(round(number * n_stride))
    print(f"Segment: start={start}, n_stride={n_stride}, start_index={start_index}, end_index={end_index}, len(eeg)={len(eeg)}")
    if end_index > len(eeg):
        print("Returning empty segment!")
        return np.array([])
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
    eeg_hi = lfilter(B, A, eeg)

    psd = np.full((N, int(4 * n_stride / 2)), np.nan)
    suppression_filter = piecewise(np.arange(0, 64, 0.5), [0, 3, 6], [0, 0.25, 1]) ** 2
    components = np.full((N, 3), np.nan)

    valid_count = 0

    for n in range(N):
        if is_not_burst_suppressed(BSRmap, n, 4):
            seg_hi = segment(eeg_hi, n + 4, 4, n_stride)
            psd[n, :] = power_spectral_density(seg_hi)

            seg_raw = segment(eeg, n + 4, 4, n_stride)
            if sawtooth_detector(seg_raw, n_stride):
                psd[n, :] *= suppression_filter

            if np.isnan(psd[n, :]).all():
                print(f"Epoch {n}: psd is all NaNs")
            else:
                valid_count += 1

        thirty_sec = time_range(30, n, stride)

        try:
            vhigh_band = band_range(39.5, 46.5, 0.5)
            vhigh_band_alt = band_range(40, 47, 0.5)
            whole_band = band_range(0.5, 46.5, 0.5)
            whole_band_alt = band_range(1, 47, 0.5)
            mid_band = band_range(11, 20, 0.5)

            # Checks if the thirty second index range is empty, skips current iteration if so
            if len(thirty_sec) == 0 or np.isnan(psd[thirty_sec][:, mid_band]).all():
                print(f"Epoch {n}: Empty slice")
                continue  # skip this epoch

            if np.isnan(psd[thirty_sec][:, vhigh_band]).all():
                print(f"Epoch {n}: vhigh_band slice all NaN, thirty_sec={thirty_sec}, vhigh_band={vhigh_band}")
            if np.isnan(psd[thirty_sec][:, whole_band]).all():
                print(f"Epoch {n}: whole_band slice all NaN, thirty_sec={thirty_sec}, whole_band={whole_band}")
            # print(f"psd[thirty_sec].shape={psd[thirty_sec].shape}")

            # Ensure PSD slices are arrays
            vhigh_slice1 = np.asarray(psd[thirty_sec][:, vhigh_band])
            vhigh_slice2 = np.asarray(psd[thirty_sec][:, vhigh_band_alt])
            whole_slice1 = np.asarray(psd[thirty_sec][:, whole_band])
            whole_slice2 = np.asarray(psd[thirty_sec][:, whole_band_alt])

            # Diagnostic Statements
            if np.isnan(psd[thirty_sec][:, vhigh_band]).all() or np.isnan(psd[thirty_sec][:, whole_band]).all():
                print(f"Epoch {n}: Component 1 slices fully NaN")

            if np.isnan(psd[thirty_sec][:, band_range(0.5, 4, 0.5)]).all():
                print(f"Epoch {n}: Component 2 low band slice fully NaN")

            # Defensive checks
            if vhigh_slice1.shape != vhigh_slice2.shape or whole_slice1.shape != whole_slice2.shape:
                raise ValueError(f"Mismatched PSD slice shapes: {vhigh_slice1.shape} vs {vhigh_slice2.shape}")

            # Compute geometric means safely
            vhigh = np.sqrt(np.nanmean(vhigh_slice1 * vhigh_slice2, axis=1))
            whole = np.sqrt(np.nanmean(whole_slice1 * whole_slice2, axis=1))

            # Safe ratio with divide
            ratio = np.divide(vhigh, whole, out=np.full_like(vhigh, np.nan), where=whole != 0)
            safe_ratio = np.maximum(ratio, 1e-8)
            log_ratios = 10 * np.log10(safe_ratio)

            # DEBUG: Print diagnostic info for component[1]
            # if n%100 == 0:
            #     print(f"\nEpoch {n}")
            #     print(f"vhigh_slice1 mean: {np.nanmean(vhigh_slice1):.4f}")
            #     print(f"vhigh_slice2 mean: {np.nanmean(vhigh_slice2):.4f}")
            #     print(f"whole_slice1 mean: {np.nanmean(whole_slice1):.4f}")
            #     print(f"whole_slice2 mean: {np.nanmean(whole_slice2):.4f}")
            #     print(f"ratio mean: {np.nanmean(ratio):.4f}")
            #     print(f"log10(ratio): {np.nanmean(10 * np.log10(np.maximum(ratio, 1e-8))):.2f} dB")
            
            mid_psd_slice = psd[thirty_sec][:, mid_band]

            if mid_psd_slice.size == 0 or np.isnan(mid_psd_slice).all():
                print(f"Skipping epoch {n} due to empty or all-NaN mid_psd_slice")
                continue  # Skip this epoch

            mid_power = prctmean(np.nanmean(10 * np.log10(np.maximum(mid_psd_slice, 1e-8)), axis=0), 50, 100)

            components[n, 0] = mean_band_power(psd[thirty_sec], 30, 47, 0.5) - mid_power

            if np.any(np.isnan(vhigh)) or np.any(np.isnan(whole)):
                print(f"Epoch {n}: NaNs in vhigh or whole slices → skipping Component 1")
                continue

            if np.any(np.isnan(ratio)):
                print(f"Epoch {n}: NaNs in ratio → skipping Component 1")
                continue

            log_ratio = 10 * np.log10(safe_ratio)
            if np.any(np.isnan(log_ratio)):
                print(f"Epoch {n}: NaNs in log10 ratio → skipping Component 1")
                continue

            # Avoid trim_mean if too few values
            if log_ratios.size >= 2:
                components[n, 1] = trim_mean(log_ratios, 0.2)
            else:
                components[n, 1] = np.nanmean(log_ratios)
                        
            components[n, 2] = mean_band_power(psd[thirty_sec], 0.5, 4, 0.5) - mid_power

        except Exception as e:
            print(f"Exception in epoch {n}: {e}")
            pass  # Handle NaNs or range issues gracefully

        # plt.plot(10 * np.log10(psd[n, :]))

        # plt.plot(vhigh, label="vhigh")
        # plt.plot(whole, label="whole")
        # plt.plot(10 * np.log10(np.maximum(vhigh / whole, 1e-8)), label="log-ratio")
        # plt.legend()
        # plt.title("Component 1 inputs")
        # plt.show()
    print(f"Valid PSDs: {valid_count}/{N}")

    return components

# Calculate the Power Spectral Density (PSD)
def power_spectral_density(x):
    x = x - baseline(x)
    win = np.blackman(len(x))
    f = fft(win * x)
    f = f[1:len(x)//2+1]  # Match MATLAB's 1-based indexing and skip bin 0
    return 2 * np.abs(f)**2 / (len(x) * np.sum(win**2))

# Sawtooth detection for K-complexes
def sawtooth_detector(eeg, n_stride):
    saw = np.concatenate([np.zeros(n_stride - 5), np.arange(1, 6)])
    saw = (saw - np.mean(saw)) / np.std(saw, ddof=0)

    conv_len = len(eeg) - len(saw) + 1
    if conv_len <= 0:
        return False  # EEG segment too short

    # Compute variance over sliding windows
    v = np.array([np.var(eeg[i:i+len(saw)], ddof=0) for i in range(conv_len)])

    # Avoid division by zero or invalid values
    v = np.where((v == 0) | np.isnan(v), np.nan, v)

    # Convolutions with forward and reversed sawtooth
    conv1 = np.convolve(eeg, saw[::-1], mode='valid')
    conv2 = np.convolve(eeg, saw, mode='valid')
    m = (np.stack([conv1, conv2]) / len(saw))**2  # shape (2, conv_len)

    # Guard against all-NaN case
    if np.isnan(v).all():
        return False

    # Compute normalized match score where variance is valid
    with np.errstate(invalid='ignore', divide='ignore'):
        m_ratio_0 = np.where(v > 10, m[0] / v, 0)
        m_ratio_1 = np.where(v > 10, m[1] / v, 0)
        m_ratio = np.maximum(m_ratio_0, m_ratio_1)

    # Optional debug print
    # print("Max m_ratio:", np.nanmax(m_ratio))

    return np.nanmax(m_ratio) > 0.63

# Band range for frequency bands
def band_range(low, high, binsize):
    return np.arange(int(low / binsize), int(high / binsize) + 1)

# Mean power in a frequency band
def mean_band_power(psd, fmin, fmax, bin_width):
    band = band_range(fmin, fmax, bin_width)
    v = psd[:, band]
    if np.isnan(v).all():
        return np.nan
    return np.nanmean(10 * np.log10(v + 1e-8))

# Check if an epoch is burst-suppressed
def is_not_burst_suppressed(BSRmap, n, p, threshold=0.5):
    # If not enough data points yet, be conservative and return False (cannot confirm no suppression)
    if n < p:
        return False

    # Extract segment from BSRmap, same indexing as MATLAB: n-p+1 to n inclusive
    segment = BSRmap[n - p + 1 : n + 1]

    # If all values are NaN, cannot decide → return False
    if np.isnan(segment).all():
        return False

    # Filter out NaNs for decision by ignoring them
    valid_values = segment[~np.isnan(segment)]

    # If no valid values remain after removing NaNs, return False
    if len(valid_values) == 0:
        return False

    # Return True if all valid values are below the threshold (not burst suppressed)
    return np.all(valid_values < threshold)

# Time range for a given number of seconds before the current epoch
def time_range(seconds, n, stride):
    start = int(max(0, np.ceil(n - seconds / stride + 1)))
    return np.arange(start, n + 1)

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

# Piecewise linear interpolation function
def piecewise(x, xp, yp):
    x = np.array(x)
    xp = np.array(xp)
    yp = np.array(yp)
    return np.interp(bound(x, xp[0], xp[-1]), xp, yp)

# Logistic S-curve function
def scurve(x, Eo, Emax, x50, xwidth):
    x = np.array(x)
    return Eo - Emax / (1 + np.exp((x - x50) / xwidth))

# Bound the values between a lower and upper bound
def bound(x, lower, upper):
    return np.clip(x, lower, upper)
