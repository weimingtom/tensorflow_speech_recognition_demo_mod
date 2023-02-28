# tensorflow_speech_recognition_demo_mod
My mod of tensorflow_speech_recognition_demo

## Original sources  
* https://github.com/llSourcell/tensorflow_speech_recognition_demo  

## Dependencies  
* use Baidu AI Studio (BML Codelab 1.7.2)    
* Python 3.7.4  
* librosa 0.7.2  
* tensorflow 2.11.0  
* tflearn 0.5.0  
* scikit-image 0.19.3  
* scikit-learn 0.22.1  

## How to run for Baidu AIStudio   
```
$ make clean
$ make
$ make test
```
```
traning, about 1 houre 16 minutes
Training Step: 1200  | total loss: 1.03208 | time: 6.489s
| Adam | epoch: 300 | loss: 1.03208 - acc: 0.7503 | val_loss: 0.13307 - val_acc: 0.9766 -- iter: 256/256
```
```
test result:  
...
start train... 2023-02-27 12:44:44
...
[[3.5936679e-04 5.3023919e-03 1.0024645e-02 2.0324470e-02 3.8327340e-03
  6.7243548e-03 7.7842027e-03 4.1595562e-03 9.3923086e-01 2.2574777e-03]]
Digit predicted:  8
the file is 8_Samantha_300.wav : result  is 8
end test... 2023-02-27 12:44:48
```

## Original README  

## tensorflow_speech_recognition_demo
This is the code for 'How to Make a Simple Tensorflow Speech Recognizer' by @Sirajology on Youtube

Overview
============
This is the full code for 'How to Make a Simple Tensorflow Speech Recognizer' by @Sirajology on [Youtube](https://youtu.be/u9FPqkuoEJ8).
In this demo code we build an LSTM recurrent neural network using the TFLearn high level Tensorflow-based library to train
on a labeled dataset of spoken digits. Then we test it on spoken digits. 

Dependencies
============
* tflearn (http://tflearn.org/)
* tensorflow  (https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html)
* future

Use [pip](https://pypi.python.org/pypi/pip) to install any missing dependencies

Usage
===========

Run the following code in terminal. This will take a couple hours to train fully.

`python demo.py`


Challenge
===========

The weekly challenge is from the last video, it's still running! Check it out [here](https://www.youtube.com/watch?v=mGYU5t8MO7s)

Credits
===========
Credit for the vast majority of code here goes to [pannouse](https://github.com/pannous). I've merely created a wrapper to get people started!
