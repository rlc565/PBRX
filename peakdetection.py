import numpy as np

import processing

def preprocess(data, window_size, sample_rate):
    # bandpass filter
    filterd_data = processing.bandpass(data, sample_rate, 15, 5)

    # Derivative and Squaring Function
    sqr_diff = np.diff(filterd_data) **2

    # Moving-window Integration
    return processing.moving_window_integration(sqr_diff[int(window_size*2):], window_size)

def pan_tompkins_beat_detection(dataset, sample_rate=360, max_beat=0.08):
    window_size = max_beat * sample_rate
    window_integration = preprocess(dataset["'MLII'"], window_size, sample_rate)

    spki = 0 # running estimate of siganl peak
    npki = 0 # running estimate of noise peak
    threshold1 = npki +0.25*(spki-npki)
    threshold2 = threshold1/2
    peaks = []
    signal_peaks = [0]
    times = []
    noise_peaks = []
    missed = 0
    index = 0

    for i in range(2, len(window_integration)-1):
        if window_integration[i-1] < window_integration[i] and window_integration[i] > window_integration[i+1]:
            peaks.append(i)

            if window_integration[i] > threshold1 and (i - signal_peaks[-1]) > 0.3*sample_rate:
                signal_peaks.append(i)
                times.append(dataset["'Elapsed time'"].iloc[int(window_size*2)+i])
                spki = 0.125*window_integration[signal_peaks[-1]] + 0.875*spki
                if missed > 0:
                    if (signal_peaks[-1] - signal_peaks[-2]) > missed:
                        missed_peaks = peaks[signal_peaks[-2]+1:signal_peaks[-1]]
                        missed_peaks2 = []
                        for missed_peak in missed_peaks:
                            if ((missed_peak - signal_peaks[-2]) > sample_rate/4
                                    and (signal_peaks[-1] - missed_peak) > sample_rate/4
                                    and window_integration[missed_peak] > threshold2):
                                missed_peaks2.append(missed_peak)

                        if len(missed_peaks2) > 0:
                            missed_peak = missed_peaks2[np.argmax(window_integration[missed_peaks2])]
                            missed_peaks.append(missed_peak)
                            signal_peaks.append(signal_peaks[-1])
                            times.append(dataset["'Elapsed time'"].iloc[int(window_size*2)+missed_peak])
                            signal_peaks[-2] = missed_peak

                else:
                    noise_peaks.append(i)
                    npki = 0.125*window_integration[noise_peaks[-1]] + 0.875*npki

                threshold1 = npki + 0.25*(spki-npki)
                threshold2 = 0.5 * threshold1

                if len(signal_peaks)>8:
                    rr = np.diff(signal_peaks[-9:])
                    mean_rr = int(np.mean(rr))
                    missed = int(1.66*mean_rr)

                index = index+1

    signal_peaks = list(map(lambda x: x+int(window_size*2), signal_peaks))
    return signal_peaks[2:len(signal_peaks)-1], window_integration
