import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import ast
import torch
from torch.nn.utils.rnn import pad_sequence

def convert_to_forward_slashes(path):
    # Convert Windows backslash paths to forward slash paths
    return path.replace('\\', '/')

def plot_ecg(exam_index, tracings_group):
    """
    Plots the ECG tracing for a given exam index.

    Parameters
    ----------
    exam_index : int
        The index of the exam to plot.
    tracings_group : numpy.ndarray
        A tensor of ECG tracings, with shape `(N, M, 12)`, where `N` is the number of exams,
        `M` is the number of samples per exam, and `12` is the number of leads.

    Returns
    -------
    None

    Raises
    ------
    IndexError
        If `exam_index` is out of range for `tracings_group`.

    Notes
    -----
    This function plots the ECG tracing for the first lead (DI) of the specified exam index.
    The x-axis of the plot shows time in seconds, and the y-axis shows amplitude in millivolts.
    The plot includes a title with the exam index.

    Examples
    --------
    >>> exam_id = 2999852
    >>> exam_index = (exam_ids == exam_id).nonzero()[0][0]
    >>> plot_ecg(exam_index, tracings_group)
    """
    # Get the tracings for the exam
    exam_tracings = tracings_group[exam_index]

    # Get the sampling frequency (in Hz)
    fs = 400

    # Calculate the time axis
    t = np.arange(exam_tracings.shape[0]) / fs

    # Plot the first lead (DI)
    plt.plot(t, exam_tracings[:, 0])
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (mV)')
    plt.title(f'ECG Tracing (Lead DI) for Exam {exam_index}')
    plt.show()

def load_and_process(filename):
    """
    Loads the CSV file and processes it into a DataFrame.
    """
    df = pd.read_csv(filename)  # Load data from the CSV file into a DataFrame
    df['dx'] = df['dx'].apply(ast.literal_eval)  # Convert 'dx' column from string representation of list to actual list
    return df  # Return the processed DataFrame

def map_codes_to_dx(codes):
    return [codes_dict.get(int(code), code) for code in codes]

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def collate_fn(batch):
    """
    Processes a batch of data by separating the texts and ECGs, and padding the ECG sequences.

    Parameters
    ----------
    batch : list of tuples
        The batch data. Each item in the batch is a tuple where the first element is the text and the second element is the ECG.

    Returns
    -------
    tuple
        A tuple containing two elements. The first element is a list of texts. The second element is a tensor of padded ECG sequences.

    Notes
    -----
    This function is typically used as the `collate_fn` argument to a `torch.utils.data.DataLoader`. The `collate_fn` is applied to a list of data samples to form a batch. This function separates the texts and ECGs from the batch data, pads the ECG sequences so they all have the same length, and returns the texts and padded ECGs.

    Examples
    --------
    >>> data = [("text1", np.array([1, 2, 3])), ("text2", np.array([4, 5, 6, 7]))]
    >>> texts, ecgs_padded = collate_fn(data)
    """
    texts = [item[0] for item in batch]
    ecgs = [item[1] for item in batch]
    # Pad the sequences
    ecgs_padded = pad_sequence([torch.from_numpy(ecg) for ecg in ecgs], batch_first=True)
    return texts, ecgs_padded