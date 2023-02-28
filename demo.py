from __future__ import division, print_function, absolute_import
import tflearn
import speech_data
#import tensorflow as tf
import numpy
import time
import os
import random
import librosa
# search mod
print('start train...', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
# mod, see https://github.com/llSourcell/tensorflow_speech_recognition_demo/blob/master/demo.py
#learning_rate = 0.0001
learning_rate = 0.001
#training_iters = 300000  # steps
# see https://github.com/PacktPublishing/Hands-On-Natural-Language-Processing-with-Python/blob/master/Chapter11/01_example.ipynb
training_iters = 30
batch_size = 64
#64 
#20000 #64

width = 20  # mfcc features
# mod, see https://github.com/llSourcell/tensorflow_speech_recognition_demo/blob/master/demo.py
#height = 80  # (max) length of utterance
height = 35  # (max) length of utterance
classes = 10  # digits


def get_mfcc_features(fpath):
    raw_w,sampling_rate = librosa.load(fpath,mono=True)
    mfcc_features = librosa.feature.mfcc(raw_w,sampling_rate)
    if(mfcc_features.shape[1]>height):
        mfcc_features = mfcc_features[:,0:height]
    else:
        mfcc_features=numpy.pad(mfcc_features,((0,0),(0,height-mfcc_features.shape[1])), 
                             mode='constant', constant_values=0)
    return mfcc_features

def get_batch_mfcc(fpath,batch_size=256):
    ft_batch = []
    labels_batch = []
    files = os.listdir(fpath)
    while True:
        print("Total %d files" % len(files))
        random.shuffle(files)
        for fname in files:
            if not fname.endswith(".wav"): 
                continue
            mfcc_features = get_mfcc_features(fpath+fname)  
            label = numpy.eye(10)[int(fname[0])]
            labels_batch.append(label)
            ft_batch.append(mfcc_features)
            if len(ft_batch) >= batch_size:
                yield ft_batch, labels_batch 
                ft_batch = []  
                labels_batch = []

#batch = word_batch = speech_data.mfcc_batch_generator(batch_size, height)
batch = word_batch = speech_data.mfcc_batch_generator(batch_size * 4, height)
#X, Y = next(batch)
#trainX, trainY = X, Y
#testX, testY = X, Y #overfit for now

# Network building
net = tflearn.input_data([None, width, height])
# mod, see https://github.com/llSourcell/tensorflow_speech_recognition_demo/blob/master/demo.py
#net = tflearn.lstm(net, 128, dropout=0.8)
# see https://github.com/PacktPublishing/Hands-On-Natural-Language-Processing-with-Python/blob/master/Chapter11/01_example.ipynb
net = tflearn.lstm(net, 128*4, dropout=0.5)
net = tflearn.fully_connected(net, classes, activation='softmax')
net = tflearn.regression(net, optimizer='adam', learning_rate=learning_rate, loss='categorical_crossentropy')
# Training

### add this "fix" for tensorflow version errors
#col = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)
#for x in col:
#    tf.compat.v1.add_to_collection(tf.compat.v1.GraphKeys.VARIABLES, x ) 

#model = tflearn.DNN(net, tensorboard_verbose=0, checkpoint_path='model_aaa', max_checkpoints=3)
#k = 0
#while k in range(1): #1: #training_iters #n_epoch=10
#  model.fit(trainX, trainY, n_epoch=10, validation_set=(testX, testY), show_metric=True,
#          batch_size=batch_size, snapshot_epoch=False, snapshot_step=1, run_id='model_aaa')
#  _y=model.predict(X)
# see https://github.com/PacktPublishing/Hands-On-Natural-Language-Processing-with-Python/blob/master/Chapter11/01_example.ipynb
model = tflearn.DNN(net, tensorboard_verbose=0)
while training_iters > 0:
  trainX, trainY = next(batch)
  testX, testY = trainX, trainY #next(batch)
  # https://github.com/tflearn/tflearn/blob/master/tutorials/intro/quickstart.md
  model.fit(trainX, trainY, n_epoch=10, validation_set=(testX, testY), show_metric=True, batch_size=batch_size) 
  training_iters -= 1
  print('training_iters:', training_iters, '...', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())) # about 1 minutes
model.save("tflearn.lstm.model")
print('end train...', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

#demo_file="8_Samantha_300.wav" # "8_Susan_200.wav"
#demo=speech_data.load_wav_file(speech_data.path+demo_file)
#result=model.predict([demo])
#result=numpy.argmax(result)
#print("the file is %s : result  is %d"%(demo_file,result))
#print('end test...', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
