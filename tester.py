import scipy.io
import numpy as np
from openibis_python import openibis  # Make sure openibis_python.py is in the same directory or installed as a module

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
    filepath = "/home/yourusername/upmc_work/openibis/Openibis/case18.mat" 
    eeg = load_eeg_from_mat(filepath)
    
    doa = openibis(eeg)
    
    print("Depth of Anesthesia Scores:")
    print(doa)

if __name__ == "__tester__":
    main()