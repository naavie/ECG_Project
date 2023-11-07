import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import ast

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