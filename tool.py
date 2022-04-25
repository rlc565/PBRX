import errno
import os
import csv
import time
import wfdb
import random
import pandas as pd
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import tensorflow as tf

#plot 'C:\Users\rebec\OneDrive\Documents\uni\PBRX\code\samples.csv' using 0:1

NUM_INPUTS = 200 # number of data points given to neural network
NUM_SAMPLES = 80 # number of samples taken from each class
MAX_DIFF = 10 # maximum difference between found peak and true peak to be used as input sample

class AnnotatedDataset:
    def __init__(self, record):
        self.record = record
        self.data = None
        self.peaks = None
        self.annotations = None

def get_datasets(scriptFile):
    """Get datasets from script file and returns a dictionary containing the filename along with a pandas DataFrame
    containing the files data.
    'Elapsed time'  'MLII'    'V5'"""
    datasets = {}
    if not os.path.exists(scriptFile):
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), scriptFile)

    with open("script.txt", 'r') as f:
        for line in f:
            line = line.strip()
            if line.endswith('.csv'):
                if os.path.exists(f'{line[:-4]}.atr'):
                    record = os.path.basename(line).split(".")[0]
                    datasets[record] = AnnotatedDataset(record)
                    datasets[record].data = pd.read_csv(line)
                    truePeaks = wfdb.rdann(line[:-4], 'atr').__dict__['sample']
                    annotations = wfdb.rdann(line[:-4], 'atr').__dict__['symbol']
                    print(len(truePeaks))
                    print(len(annotations))
                    datasets[record].annotations = dict(zip(truePeaks, annotations))
                else:
                    print("no equivalent atr file")
                #ann.sample shows location in sample
                #print(ann.symbol) # gets annotations
            else:
                # should log
                print("not csv:", line)

    if len(datasets) == 0:
        raise ValueError("No csv files found in script")
    return datasets

def bandpass(data, sampleRate, high, low, order=1):
    """ Apply a Butterworth filter to data

        params:
            data (pandas DataFrame) : list of ecg data
            sampleRate (int) : sample rate of the ecg recording
            high (int) : high cut off frequency
            low (int) : low cut off frequency
            order (int) : order of Butterworth filter

        return:
            filterdData (list)
    """
    nyquist = 0.5 * sampleRate
    high = high / nyquist
    low = low / nyquist
    b, a = signal.butter(order, [low, high], btype='bandpass')
    filterdData = signal.lfilter(b, a, pd.to_numeric(data[1:]))
    return filterdData

def moving_window_integration(data, windowSize):
    movingWindowIntegration = []
    movingWindowIntegration.append(data[0])

    for i in range(1, len(data)+1):
        if i < windowSize:
            window = data[0:i]
        else:
            window = data[int(i-windowSize):i]

        movingWindowIntegration.append(np.mean(window))

    return movingWindowIntegration

def pan_tompkins_beat_detection(dataset, sampleRate=360, maxBeat=0.08, threshold=0):
    """desired passband = 5-15 hz"""
    # bandpass filter
    filterdData = bandpass(dataset["'MLII'"],sampleRate,12,5)
    print(dataset["'MLII'"])
    print(len(dataset["'MLII'"]))
    print(len(filterdData))

    # Derivative and Squaring Function
    sqrDiff = np.diff(filterdData) **2

    # Moving-window Integration
    windowSize = maxBeat * sampleRate
    movingWindowIntegration = moving_window_integration(sqrDiff[int(windowSize*2):], windowSize)
    print(int(windowSize*2))
    print(len(movingWindowIntegration))
    # Fiducial Mark
    SPKI = 0 # running estimate of siganl peak
    NPKI = 0 # running estimate of noise peak
    threshold1 = NPKI +0.25*(SPKI-NPKI)
    threshold2 = threshold1/2
    peaks = []
    signalPeaks = [0]
    times = []
    noisePeaks = []
    missed = 0
    index = 0

    for i in range(2, len(movingWindowIntegration)-1):
        if movingWindowIntegration[i-1] < movingWindowIntegration[i] and movingWindowIntegration[i] > movingWindowIntegration[i+1]:
            peaks.append(i)

            if movingWindowIntegration[i] > threshold1 and (i - signalPeaks[-1]) > 0.3*sampleRate:
                signalPeaks.append(i)
                times.append(dataset["'Elapsed time'"].iloc[int(windowSize*2)+i])
                SPKI = 0.125*movingWindowIntegration[signalPeaks[-1]] + 0.875*SPKI
                if missed < 0:
                    if (signalPeaks[-1] - signalPeaks[-2]) > missed:
                        missedPeaks = peaks[signalPeaks[-2]+1:signalPeaks[-1]]
                        missedPeaks2 = []
                        for missedPeak in missedPeaks:
                            if ((missedPeak - signalPeaks[-2]) > sampleRate/4
                                    and (signalPeaks[-1] - missedPeak) > sampleRate/4
                                    and movingWindowIntegration[missedPeak] > threshold2):
                                missedPeaks2.append(missed_peak)

                        if len(missedPeaks2) > 0:
                            missedPeak = missedPeaks2[np.argmax(movingWindowIntegration[missedPeaks2])]
                            missedPeaks.append(missedPeak)
                            signalPeaks.append(signalPeaks[-1])
                            times.append(dataset["'Elapsed time'"].iloc[int(windowSize*2)+missedPeak])
                            print(missedPeak)
                            signalPeaks[-2] = missedPeak

                else:
                    noisePeaks.append(i)
                    NPKI = 0.125*movingWindowIntegration[noisePeaks[-1]] + 0.875*NPKI

                threshold1 = NPKI + 0.25*(SPKI-NPKI)
                threshold2 = 0.5*threshold1

                if len(signalPeaks)>8:
                    RR = np.diff(signalPeaks[-9:])
                    averageRR = int(np.mean(RR))
                    missed = int(1.66*averageRR)

                index = index+1

    #print(times)
    #print(noisePeaks)
    signalPeaks = list(map(lambda x: x+int(windowSize*2), signalPeaks))
    return signalPeaks[2:len(signalPeaks)-1], movingWindowIntegration

def segment(dataset, fdata, peaks):
    data = []
    labels = []
    for i in range(1, len(peaks)-1):
        for j in range(-MAX_DIFF, MAX_DIFF+1):
            if peaks[i]-14+j in dataset.annotations:
                start = (peaks[i]+j-14 - (NUM_INPUTS//2))
                end = (peaks[i]+j-14 + (NUM_INPUTS//2))
                peak = []
                for point in range(start, end):
                    peak.append(dataset.data["'MLII'"][point])
                    #peak.append(fdata[point-57+14])
                data.append([float(x) for x in peak])
                labels.append(dataset.annotations[peaks[i]-14+j])
    fig2, ax2 = plt.subplots()
    if len(data) > 0:
        x = [float(point) for point in data[0]]
        line2 = ax2.plot(x)
        ax2.set_xlabel("Index")
        ax2.set_ylabel("Frequency (Hz)")
        plt.title(label=f"Unprocessed heartbeat")
    return list(zip(data, labels))

def format_input(dataset):
    data = []
    cols = []
    for peak in dataset:
        peak = ([(x+2)/4 for x in peak[0]], peak[1])
        if peak[1] == 'N':
            input = peak[0] + [0]
        else:
            input = peak[0] + [1]
        #print(input)
        data.append(input)
    for i in range(NUM_INPUTS):
        cols.append(str(i))
    cols.append('labels')
    df = pd.DataFrame(data, columns = cols)
    return df

def tensorflow_ml(trainingData, testData):
    # takes training dataset of a dataframe of peaks data
    dataset = format_input(trainingData)
    data = dataset.drop(labels='labels', axis=1)
    labels = dataset['labels']
    print(data.shape)

    testDataset = format_input(testData)
    testData = testDataset.drop(labels='labels', axis=1)
    testLabels = testDataset['labels']

    data = np.array(data)
    testData = np.array(testData)

    print(data[0])
    print(labels[0])

    #normalise = layers.Normalization()
    #normalise.adapt(data)

    model = tf.keras.Sequential([
      #normalise,
      tf.keras.layers.Dense(64, activation='relu'),
      #tf.keras.layers.Dense(32, activation='relu'),
      # 2 outputs as one of 2 classes
      tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(loss = tf.losses.MeanSquaredError(),
                    optimizer = tf.optimizers.Adam(lr=1e-6),
                    metrics=['accuracy'])

    model.fit(data, labels, epochs=10)

    print(model.summary())

    test_loss, test_acc = model.evaluate(testData,  testLabels, verbose=2)

    print('\nTest accuracy:', test_acc)

    probability_model = tf.keras.Sequential([model,
                                         tf.keras.layers.Softmax()])

    predictions = probability_model.predict(testData)

    pred = []
    for prediction in predictions:
        pred.append(np.argmax(prediction))

    print(pred)
    #predictions = model(data).np()
    #print(predictions)
    #tf.nn.softmax(predictions).np()

def run_tool():
    datasets = get_datasets("script.txt")

    #plot_graphs(datasets)
    #x, fdata = pan_tompkins_beat_detection(datasets['100excel'].data)#.head(30000))
    x, y = generate_data(datasets)

    trainingData = x[:int(NUM_SAMPLES*0.8)] + y[:int(NUM_SAMPLES*0.8)]
    testingData = x[int(NUM_SAMPLES*0.8):] + y[int(NUM_SAMPLES*0.8):]
    random.shuffle(trainingData)
    random.shuffle(testingData)
    #print(x,y)
    tensorflow_ml(trainingData, testingData)
    # xi + (windowSize*2) - 13 or 14
    '''
    data = []
    for i in x:
        data.append(i - 16)
        #data.append(i + 114)
    '''
    #print(data)

    #plot_graphs(fdata)

    plt.show()

def generate_data(datasets):
    totalData = []
    normalData = []
    abnormalData = []

    for dataset in datasets:
        x, fdata = pan_tompkins_beat_detection(datasets[dataset].data)
        #for peak in segment(datasets[dataset], fdata, x):
        for peak in segment(datasets[dataset], datasets[dataset].data, x):
            totalData.append(peak)
    #print(totalData)
    random.shuffle(totalData)
    for item in totalData:
        if item[1] == 'N' and len(normalData) < 1000:# int(NUM_SAMPLES):
            normalData.append(item)
        elif item[1] != 'N' and len(abnormalData) < 1000:# int(NUM_SAMPLES):
            abnormalData.append(item)

    print(len(normalData))
    print(len(abnormalData))
    return normalData, abnormalData

def output_results():
    pass

def generate_graphs():
    """set datafile separator “,”
    set autoscale fix
    set key outside right center

    set title 'Title'
    set ylabel '<yLabel>'
    set xlabel '<xLabel>'

    set term png
    set output "graph.png" //graph.png must exist

    plot "C:/Users/rebec/OneDrive/Documents/uni/PBRX/code/ECGs.csv" using 2 title “ECG” with lines
    """
    subprocess.run()

def plot_graphs(datasets):
    for record,data in datasets.items():
        fig2, ax2 = plt.subplots()
        line2 = ax2.plot(data.data.head(1000)["'Elapsed time'"], data.data.head(1000)["'MLII'"])
        ax2.set_xlabel("Time elapsed(s)")
        ax2.set_ylabel("Frequency (value)")
        plt.title(label=f"{record}")

run_tool()
