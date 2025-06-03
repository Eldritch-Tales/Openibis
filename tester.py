import scipy.io
import numpy as np
from openibis import compute_depth_of_anesthesia  # Make sure openibis.py is in the same directory or installed as a module

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
    filepath = "/home/yourusername/Downloads/data.mat"  # Replace with your actual path
    eeg = load_eeg_from_mat(filepath)
    
    doa = compute_depth_of_anesthesia(eeg)
    
    print("Depth of Anesthesia Scores:")
    print(doa)

if __name__ == "__tester__":
    main()