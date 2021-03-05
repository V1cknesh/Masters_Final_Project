import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.linear_model
import glob
import gc
import keras
import tensorflow as tf
from keras.utils import np_utils
from zlib import crc32
import multiprocessing as mp
import os
import random
from DeepFingerprinting import DeepFingerprintingNeuralNetwork

threshold="0.09"
social_media_site="fb"


rootdir = '/home/student/MachineLearningTest/Masters_Final_Project/dataset/youtube/0.1_data'

SOURCE_IPADDRESS = ['172.31.40', '172.31.47', '172.31.36', '172.31.46', '172.31.33']

def ipaddressconverter(ipaddress): 
    return 1 if '.'.join(ipaddress.split('.')[0:-1]) in SOURCE_IPADDRESS  else -1;

training = []
testing = []
count = 0
for subdir, dirs, files in os.walk(rootdir):
    if (subdir.split("_")[-1] == "data"):
        threshold=subdir.split("/")[-1].split("_")[0] 
        print(threshold)
        for subdirs, dirs, files in os.walk(subdir):
            for file in files:
                filename = subdir + "/" + file
                if file.split("_")[0] == "training":
                    df = pd.read_csv(filename, index_col=0)[1:]
                    df['PACKET_SIZE'] = pd.to_numeric(df['PACKET_SIZE'], errors='coerce').astype('float32')
                    df['TOTAL_PACKET_SIZE'] = pd.to_numeric(df['TOTAL_PACKET_SIZE'], errors='coerce').astype('float32')
                    df['CUMULATIVE_PACKET_SIZE'] = pd.to_numeric(df['CUMULATIVE_PACKET_SIZE'], errors='coerce').astype('float32')
                    df['TIME'] = pd.to_numeric(df['TIME'], errors='coerce').astype('float32')
                    df['SRC'] = pd.to_numeric(df['SRC'].transform(func=lambda x: ipaddressconverter(x)), errors='coerce').astype('float32')
                    value = random.randint(0,1)
                    training.append(df)
                else:
                    tf = pd.read_csv(filename, index_col=0)[1:]
                    tf['PACKET_SIZE'] = pd.to_numeric(tf['PACKET_SIZE'], errors='coerce').astype('float32')
                    tf['TOTAL_PACKET_SIZE'] = pd.to_numeric(tf['TOTAL_PACKET_SIZE'], errors='coerce').astype('float32')
                    tf['CUMULATIVE_PACKET_SIZE'] = pd.to_numeric(tf['CUMULATIVE_PACKET_SIZE'], errors='coerce').astype('float32')
                    tf['TIME'] = pd.to_numeric(tf['TIME'], errors='coerce').astype('float32')
                    tf['SRC'] = pd.to_numeric(tf['SRC'].transform(func=lambda x: ipaddressconverter(x)), errors='coerce').astype('float32')
                    testing.append(tf)
            final_training = pd.concat(training,axis=0,ignore_index=True)[:-1]
            final_testing = pd.concat(testing,axis=0,ignore_index=True)[:-1]
            final_training[['TIME','SRC','PACKET_SIZE','CUMULATIVE_PACKET_SIZE','TOTAL_PACKET_SIZE']].to_csv("X_training"+"-"+str(threshold)+".csv", sep=",", encoding='utf-8')
            final_training['PAGE_NUMBER'].to_csv("Y_training"+"-"+str(threshold)+".csv", sep=",", encoding='utf-8')
            final_testing[['TIME','SRC','PACKET_SIZE','CUMULATIVE_PACKET_SIZE','TOTAL_PACKET_SIZE']].to_csv("X_testing"+"-"+str(threshold)+".csv", sep=",", encoding='utf-8')
            final_testing['PAGE_NUMBER'].to_csv("Y_testing"+"-"+str(threshold)+".csv", sep=",", encoding='utf-8')



random.seed(0)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

X_train = pd.read_csv('/home/student/MachineLearningTest/Masters_Final_Project/X_training-0.1.csv', index_col=0).to_numpy()
y_train = pd.read_csv('/home/student/MachineLearningTest/Masters_Final_Project/Y_training-0.1.csv', index_col=0)
X_valid = pd.read_csv('/home/student/MachineLearningTest/Masters_Final_Project/X_testing-0.1.csv', index_col=0).to_numpy() 
y_valid = pd.read_csv('/home/student/MachineLearningTest/Masters_Final_Project/Y_testing-0.1.csv', index_col=0)


# we need a [Length x 1] x n shape as input to the DFNet (Tensorflow)
X_train = X_train[:, :,np.newaxis]
X_valid = X_valid[:, :,np.newaxis]
INPUT_SHAPE = (5,1)

NUMBER_OF_PAGES=101

#NUMBER_OF_PAGES=97
# convert class vectors to binary class matrices
y_train = np_utils.to_categorical(y_train['PAGE_NUMBER'].to_numpy())
y_valid = np_utils.to_categorical(y_valid['PAGE_NUMBER'].to_numpy())
#DeepFingerprinting Steps
model = DeepFingerprintingNeuralNetwork.neuralnetwork(input=INPUT_SHAPE, N=NUMBER_OF_PAGES)
model.fit(X_train, y_train, batch_size=100,shuffle=True, epochs=3, verbose=1, validation_data=(X_valid, y_valid))
X_valid = X_valid.astype('float32')
X_valid = X_valid[:, :,np.newaxis]
result = model.predict(X_valid, verbose=2)

TP = 0
labels = np
FP = 0
TN = 0
FN = 0
for i in range(len(result)):
    sm_vector = result[i]
    predicted_class = np.argmax(sm_vector)
    max_prob = max(sm_vector)
    if max_prob >= 0.8: # predicted as Monitored and actual site is Monitored
        TP = TP + 1
    else: # predicted as Unmonitored and actual site is Monitored
        FP = FP + 1

# ==============================================================
# Test with Unmonitored testing instances
# evaluation
for i in range(len(result)):
    sm_vector = result[i]
    predicted_class = np.argmax(sm_vector)
    max_prob = max(sm_vector)

    #if max_prob >= 0.8: # predicted as Monitored and actual site is Unmonitored
        #FP = FP + 1
    #else: # predicted as Unmonitored and actual site is Unmonitored
        #TN = TN + 1

print("TP : ", TP)
print("FP : ", FP)
print("True positive rate : ", TP / (TP + FN))
print("False positive rate : ", FP / (TP + TN))
print("Accuracy  : ", (TP + TN) / len(y_valid))

gc.collect()






