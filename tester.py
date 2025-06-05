import scipy.io
import numpy as np
from openibis_python import openibis  # Make sure openibis_python.py is in the same directory or installed as a module
import matplotlib.pyplot as plt


def load_eeg_from_mat(filepath, variable_name='eeg'):
    """
    Load EEG data from a .mat file.
    Assumes the EEG data is stored in a variable called 'eeg'.
    """
    mat = scipy.io.loadmat(filepath)
    if variable_name not in mat:
        raise KeyError(f"Variable '{variable_name}' not found in {filepath}")
    eeg = mat[variable_name]
    return eeg.squeeze()  # Remove extra dimensions if present

def main():
    filepath = "/home/sriram/upmc_work/openibis/Openibis/case7.mat" 
    # eeg = load_eeg_from_mat(filepath)
    
    
    doa = openibis(filepath)
    doa = np.nan_to_num(doa, nan=0.0, posinf=0.0, neginf=0.0)

    # print("Depth of Anesthesia Scores:")
    # print(doa)
    Fs = 128
    stride = 0.5  # from your code
    time = np.arange(len(doa)) * stride

    # Plot the depth of anesthesia
    plt.figure(figsize=(12, 4))
    plt.plot(time, doa, label="Depth of Anesthesia", color="blue")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Score")
    plt.title("Estimated Depth of Anesthesia Over Time")
    plt.grid(True)
    plt.ylim(0, 100)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()