import errno
import os
import random
import wfdb
import pandas as pd
import matplotlib.pyplot as plt

import peakdetection
import nn

NUM_INPUTS = 150 # number of data points given to neural network
NUM_SAMPLES = 71 # number of samples taken from each class
MAX_DIFF = 10 # maximum difference between found peak and true peak to be used as input sample

class AnnotatedDataset:
    def __init__(self, record):
        self.record = record
        self.data = None
        self.peaks = None
        self.annotations = None

def get_datasets(script_file):
    """Get datasets from script file and returns a dictionary containing the filename along with a pandas DataFrame
    containing the files data.
    """
    datasets = {}
    if not os.path.exists(script_file):
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), script_file)

    with open("script.txt", 'r') as file:
        for line in file:
            line = line.strip()
            if line.endswith('.csv'):
                if os.path.exists(f'{line[:-4]}.atr'):
                    record = os.path.basename(line).split(".")[0]
                    datasets[record] = AnnotatedDataset(record)
                    datasets[record].data = pd.read_csv(line)
                    true_peaks = wfdb.rdann(line[:-4], 'atr').__dict__['sample']
                    annotations = wfdb.rdann(line[:-4], 'atr').__dict__['symbol']
                    datasets[record].annotations = dict(zip(true_peaks, annotations))
                else:
                    print("no equivalent atr file")
            else:
                print("not csv:", line)

    if len(datasets) == 0:
        raise ValueError("No csv files found in script")
    return datasets

def segment(dataset, peaks):
    ''' Split data around the R-peak with NUM_INPUTS datapoints either side of each of the peaks
    indexed in peaks and returns a list of tuples of the peak and the quivalent annotation.
    '''
    data = []
    labels = []
    for i in range(2, len(peaks)-1):
        for j in range(-MAX_DIFF, MAX_DIFF+1):
            if peaks[i]-14+j in dataset.annotations:
                start = (peaks[i]+j-14 - (NUM_INPUTS//2))
                end = (peaks[i]+j-14 + (NUM_INPUTS//2))
                peak = []
                for point in range(start, end):
                    peak.append(dataset.data["'MLII'"][point])

                data.append([float(x) for x in peak])
                labels.append(dataset.annotations[peaks[i]-14+j])

                #plot_graph(peak, f"Unprocessed heartbeat {dataset.annotations[peaks[i]-14+j]}")

    return list(zip(data, labels))

def run_tool():
    datasets = get_datasets("script.txt")

    x, y = generate_data(datasets)

    training_data = x[:int(NUM_SAMPLES*0.8)] + y[:int(NUM_SAMPLES*0.8)]
    testing_data = x[int(NUM_SAMPLES*0.8):] + y[int(NUM_SAMPLES*0.8):]
    random.shuffle(training_data)
    random.shuffle(testing_data)

    nn.tensorflow_ml(training_data, testing_data, NUM_INPUTS)

    plt.show()

def generate_data(datasets):
    ''' Get normal and abnormal peaks with the number of each given by NUM_SAMPLES.'''
    total_data = []
    normal_data = []
    abnormal_data = []

    for dataset in datasets:
        x, _ = peakdetection.pan_tompkins_beat_detection(datasets[dataset].data)

        print(f'Peaks for sample {datasets[dataset].record}: {x}')
        
        for peak in segment(datasets[dataset], x):
            total_data.append(peak)

    random.shuffle(total_data)
    for item in total_data:
        if item[1] == 'N' and len(normal_data) < int(NUM_SAMPLES):
            normal_data.append(item)
        elif item[1] != 'N' and len(abnormal_data) < int(NUM_SAMPLES):
            abnormal_data.append(item)

    return normal_data, abnormal_data

def plot_graph(data, title, xlabel="Index", ylabel="Frequency (Hz)"):
    ''' Plot a heartbeat.'''
    _, ax2 = plt.subplots()
    if len(data) > 0:
        x = [float(x) for x in data]
        _ = ax2.plot(x)
        ax2.set_xlabel(xlabel)
        ax2.set_ylabel(ylabel)
        plt.title(title)

run_tool()
