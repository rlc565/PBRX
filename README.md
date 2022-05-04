# ECG Peak Detection Algorithm and Classifying Neural Network

## Intro

This code runs an implementation of the Pan and Tompkins peak detection algorithm [1] on ECG data from the [MIT-BIH ahrythmia database](https://physionet.org/content/mitdb/1.0.0/) [2]. It then segments the data around these peaks with a default of 75 datapoints either side of the peak. This can be changed by altering 'NUM_INPUTS' in the 'tool.py' file. It then feeds these samples into a neural network to train in to classify each heartbeat as normal or abnormal.

## Dependencies

The following modules will need to be installed to run the code:
* matplotlib
* numpy
* pandas
* scipy
* tensorflow
* wfdb

## How to run

CSV data should be taken From the [MIT-BIH ahrythmia database](https://physionet.org/content/mitdb/1.0.0/) [2]. These files should then be listed in the 'script.txt' file.

Make sure all CSV files listed within 'script.txt' and their equivalent ATR files are in the same folder as the code.

Then the code can be run by running the 'tool.py' file.

This will output the indexes of all the peak found in each of the CSV files as well as the acurracies achived training the neural network and the neural network's predictions for the classes of the testing data along with thier true classifications. A classification of 1 shows an abnormal heartbeat and a classification of 0 shows a normal heartbeat.

## References

[1] J. Pan and W. J. Tompkins, "A Real-Time QRS Detection Algorithm," IEEE Transactions on Biomedical Engineering, vol. 32, no. 3, pp. 230-236, March 1985. doi: 10.1109/TBME.1985.325532

[2] G. B. Moody and R. G. Mark, “The impact of the MIT-BIH Arrhythmia Database,” IEEE Eng in Med and Biol, vol. 20, no. 3, pp. 45-50, May-June 2001.
