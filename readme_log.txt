about 12:21-11:05=1 houre 16 minutes


start train... 2023-02-27 11:05:54
...
---------------------------------
Run id: UPCQA6
Log directory: /tmp/tflearn_logs/
---------------------------------
Training samples: 256
Validation samples: 256
--
Training Step: 1164  | total loss: 1.15965 | time: 7.141s
| Adam | epoch: 291 | loss: 1.15965 - acc: 0.7203 | val_loss: 0.28688 - val_acc: 0.9180 -- iter: 256/256
--
Training Step: 1168  | total loss: 1.12608 | time: 6.370s
| Adam | epoch: 292 | loss: 1.12608 - acc: 0.7194 | val_loss: 0.21541 - val_acc: 0.9453 -- iter: 256/256
--
Training Step: 1172  | total loss: 0.94108 | time: 6.398s
| Adam | epoch: 293 | loss: 0.94108 - acc: 0.7589 | val_loss: 0.23047 - val_acc: 0.9453 -- iter: 256/256
--
Training Step: 1176  | total loss: 1.06346 | time: 6.499s
| Adam | epoch: 294 | loss: 1.06346 - acc: 0.7336 | val_loss: 0.19514 - val_acc: 0.9727 -- iter: 256/256
--
Training Step: 1180  | total loss: 1.09535 | time: 6.299s
| Adam | epoch: 295 | loss: 1.09535 - acc: 0.7304 | val_loss: 0.18999 - val_acc: 0.9648 -- iter: 256/256
--
Training Step: 1184  | total loss: 1.05173 | time: 7.105s
| Adam | epoch: 296 | loss: 1.05173 - acc: 0.7379 | val_loss: 0.16490 - val_acc: 0.9727 -- iter: 256/256
--
Training Step: 1188  | total loss: 1.04942 | time: 6.888s
| Adam | epoch: 297 | loss: 1.04942 - acc: 0.7387 | val_loss: 0.16210 - val_acc: 0.9766 -- iter: 256/256
--
Training Step: 1192  | total loss: 0.84206 | time: 6.601s
| Adam | epoch: 298 | loss: 0.84206 - acc: 0.7909 | val_loss: 0.15205 - val_acc: 0.9844 -- iter: 256/256
--
Training Step: 1196  | total loss: 0.83802 | time: 6.506s
| Adam | epoch: 299 | loss: 0.83802 - acc: 0.7928 | val_loss: 0.13596 - val_acc: 0.9805 -- iter: 256/256
--
Training Step: 1200  | total loss: 1.03208 | time: 6.489s
| Adam | epoch: 300 | loss: 1.03208 - acc: 0.7503 | val_loss: 0.13307 - val_acc: 0.9766 -- iter: 256/256
--
training_iters: 0 ... 2023-02-27 12:21:34
end train... 2023-02-27 12:21:34



-----------

aistudio@jupyter-387843-3915722:~/tensorflow_speech_recognition_demo-master$ make test
python test.py  
...
start train... 2023-02-27 12:44:44
...
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2023-02-27 12:44:46.881986: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:357] MLIR V1 optimization pass is not enabled
[[3.5936679e-04 5.3023919e-03 1.0024645e-02 2.0324470e-02 3.8327340e-03
  6.7243548e-03 7.7842027e-03 4.1595562e-03 9.3923086e-01 2.2574777e-03]]
Digit predicted:  8
the file is 8_Samantha_300.wav : result  is 8
end test... 2023-02-27 12:44:48

