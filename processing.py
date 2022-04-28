import pandas as pd
import numpy as np
from scipy import signal

def bandpass(data, sample_rate, high, low, order=1):
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
    nyquist = 0.5 * sample_rate
    high = high / nyquist
    low = low / nyquist
    b, a = signal.butter(order, [low, high], btype='bandpass')
    filterd_data = signal.lfilter(b, a, pd.to_numeric(data[1:]))
    return filterd_data

def moving_window_integration(data, window_size):
    window_integration = []
    window_integration.append(data[0])

    for i in range(1, len(data)+1):
        if i < window_size:
            window = data[0:i]
        else:
            window = data[int(i-window_size):i]

        window_integration.append(np.mean(window))

    return window_integration
