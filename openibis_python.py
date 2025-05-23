import numpy as np
from scipy.signal import butter, filtfilt, blackman, fftconvolve
from scipy.fft import fft
from scipy.stats import trim_mean
from scipy.ndimage import uniform_filter1d
import math