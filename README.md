# Audio-Genre-Classifier
A song classification tool created for ECE 477: HW for Machine Learning

LSTM code deps:
-keras
-librosa
-matplotlib
-numpy
-pytorch-lightning
-tensorflow
-torch
-torchvision

to test LSTM, use:
`python3 classifier_predict.py path/to/custom/file.mp3`

LSTM code is modified version of https://github.com/ruohoruotsi/LSTM-Music-Genre-Classification/tree/master
Huge credit to https://github.com/ruohoruotsi . The code was modified to include more classifications and obviously we are doing the classification with a live recording, and the 
architecture is modified to try and achieve our goals