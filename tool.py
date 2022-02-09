import errno
import os
import csv
import pandas as pd
import numpy as np
from scipy import signal

#plot 'C:\Users\rebec\OneDrive\Documents\uni\PBRX\code\samples.csv' using 0:1

def get_datasets(scriptFile):
    """Get datasets from script file and returns a dictionary containing the filename along with a pandas DataFrame
    containing the files data."""
    datasets = {}
    if not os.path.exists(scriptFile):
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), scriptFile)

    with open("script.txt", 'r') as f:
        for line in f:
            line = line.strip()
            if line.endswith('.csv'):
                datasets[os.path.basename(line).split(".")[0]] = pd.read_csv(line)
            else:
                # should log
                print("not csv:", line)

    if len(datasets) == 0:
        raise ValueError("No csv files found in script")
    return datasets

def read_datasets2(datasets):
    ecgData = {}
    for dataset in datasets:
        tempData = {}
        data = pd.read_csv(dataset)
        print(data)


def read_datasets(datasets):
    for dataset in datasets:
        with open(dataset, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='|')
            linecount = 0
            for row in reader:
                if linecount == 0:
                    print(', '.join(row))
                else:
                    print(f'one: {row[0]}, two: {row[1]}, three {row[2]}')
                linecount += 1

def bandpass(datasets, low, high):
    # sampling frequency
    f_sample = 360
    # pass band ripple
    fs = 0.5
    # Sampling Time
    Td = 1
    # pass band ripple
    g_pass = 0.4
    # stop band attenuation
    g_stop = 50

    passband = [(2/Td)*np.tan(2*((low/2)/(f_sample/2))/2),(2/Td)*np.tan(2*((high/2)/(f_sample/2))/2)]

    order, cutoff = signal.buttord(passband, passband, g_pass, g_stop, analog=True)
    b, a = signal.butter(order, cutoff, 'bandpass', True)
    z, p = signal.bilinear(b, a, fs)
    frequency, magnitude = signal.freqz(z, p, 512)

    for dataset, ecgData in datasets:
        pass


def run_tool():
    pass

def output_results():
    pass

def generate_graphs():
    pass

print(get_datasets("script.txt"))
