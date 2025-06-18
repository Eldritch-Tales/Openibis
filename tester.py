import scipy.io
import numpy as np
from openibis_python import openibis  # Make sure openibis_python.py is in the same directory or installed as a module
import matplotlib.pyplot as plt
import h5py


def load_mat_file(filepath):
    """
    Load a .mat file (v7.0/v7.3 compatible). Returns a dictionary of variables.
    """
    try:
        # Try loading with scipy (for MATLAB v7.2 and earlier)
        data = scipy.io.loadmat(filepath)
        
        # Remove metadata keys
        data = {k: v for k, v in data.items() if not k.startswith('__')}
        print("Loaded using scipy.io.loadmat")
        return data
    
    except NotImplementedError:
        # Fall back to h5py (for MATLAB v7.3 HDF5 files)
        # print("Falling back to h5py (HDF5 format)...")
        data = {}
        with h5py.File(filepath, 'r') as f:
            def recursively_load(h5obj):
                out = {}
                for key in h5obj.keys():
                    item = h5obj[key]
                    if isinstance(item, h5py.Group):
                        out[key] = recursively_load(item)
                    else:
                        out[key] = item[()]
                return out

            data = recursively_load(f)
        return data

def main():
    filepath = "/home/sriram/upmc_work/openibis/Openibis/TestCases/case8.mat" 

    data = load_mat_file(filepath)

    # Explore keys
    print(data.keys())

    # Access EEG and BIS (example — will depend on your file)
    eeg_data = data.get("EEG", None)
    bis_data = data.get("bis", None)
    
    
    doa = openibis(eeg_data)
    doa = np.nan_to_num(doa, nan=0.0, posinf=0.0, neginf=0.0)

    print("EEG type:", type(eeg_data), "shape:", getattr(eeg_data, 'shape', 'Unknown'))
    print("BIS type:", type(bis_data), "shape:", getattr(bis_data, 'shape', 'Unknown'))
    print("DOA type:", type(doa), "shape:", getattr(doa, 'shape', 'Unknown'))

    # print("Depth of Anesthesia Scores:")
    # print(doa)
    # Fs = 128
    # stride = 0.5 
    # time = np.arange(len(doa)) * stride

    # Plot the depth of anesthesia
    # plt.figure(figsize=(12, 4))
    # plt.plot(time, doa, label="Depth of Anesthesia", color="blue")
    # plt.xlabel("Time (seconds)")
    # plt.ylabel("DOA Score")
    # plt.title("Estimated Depth of Anesthesia Over Time")
    # plt.grid(True)
    # plt.ylim(0, 100)
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

    # Create time axes
    eeg = eeg_data.squeeze()
    time_eeg = np.arange(len(eeg)) / 128  # 128 Hz
    bis = bis_data.squeeze()
    time_bis = np.linspace(0, time_eeg[-1], len(bis))
    time_doa = np.linspace(0, time_eeg[-1], len(doa))

    # Set up plot
    fig, ax1 = plt.subplots(figsize=(14, 6))

    # EEG on left y-axis
    ax1.plot(time_eeg, eeg, color='tab:blue', linewidth=0.5, label='EEG')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('EEG Amplitude', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    # BIS and Depth of Anesthesia on right y-axis
    ax2 = ax1.twinx()

    ax2.plot(time_bis, bis, color='tab:red', linewidth=2, label='BIS', alpha=0.7)
    ax2.plot(time_doa, doa.squeeze(), color='tab:green', linewidth=2, linestyle='--', label='DOA', alpha=0.7)
    ax2.set_ylabel('BIS / DOA Value', color='black')
    ax2.tick_params(axis='y', labelcolor='black')
    ax2.set_ylim(0, 100)

    # Combine legends
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')

    plt.title("EEG, BIS, and DOA Over Time")
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()