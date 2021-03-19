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
from sklearn.metrics import multilabel_confusion_matrix
from DeepFingerprinting2 import DeepFingerprintingNeuralNetwork
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from kerastuner.tuners import RandomSearch
import kerastuner as kt
from sklearn.preprocessing import label_binarize

threshold="0.09"
social_media_site="fb"


rootdir = '/home/student/MachineLearningTest/Masters_Final_Project/dataset/instagram/0.1_data'

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
                    length_of_data = len(df.index)
                    lower_quartile = int(0.15 * length_of_data)
                    upper_quartile = int(0.85 * length_of_data)
                    df.drop_duplicates(inplace=True)
                    df = df[0: len(df.index): 1]
                    df['PACKET_SIZE'] = pd.to_numeric(df['PACKET_SIZE'], errors='coerce').astype('float32')
                    df['TOTAL_PACKET_SIZE'] = pd.to_numeric(df['TOTAL_PACKET_SIZE'], errors='coerce').astype('float32')
                    df['CUMULATIVE_PACKET_SIZE'] = pd.to_numeric(df['CUMULATIVE_PACKET_SIZE'], errors='coerce').astype('float32')
                    df['TIME'] = pd.to_numeric(df['TIME'], errors='coerce').astype('float32')
                    df['SRC'] = pd.to_numeric(df['SRC'].transform(func=lambda x: ipaddressconverter(x)), errors='coerce').astype('float32')
                    training.append(df)
                else:
                    tf = pd.read_csv(filename, index_col=0)[1:]
                    length_of_data = len(tf.index)
                    lower_quartile = int(0.15 * length_of_data)
                    upper_quartile = int(0.85 * length_of_data)
                    tf.drop_duplicates(inplace=True)
                    tf = tf[0: len(tf.index): 1]
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
#tuner_search=RandomSearch(model, objective='val_accuracy', max_trials=5, directory='./output', project_name='deep_fingerprinting')
model.fit(X_train, y_train, batch_size=100,shuffle=True, epochs=1, verbose=1, validation_data=(X_valid, y_valid))



pre_cla = model.predict(X_valid, verbose=2, batch_size=128)

cm = multilabel_confusion_matrix(y_valid.argmax(axis=1), pre_cla.argmax(axis=1))
print(cm)

TN = cm[:, 0, 0]
TP = cm[:, 1, 1]
FN = cm[:, 1, 0]
FP = cm[:, 0, 1]
print("True Positive Rates")
print("*******************")
print(TP / (TP + FN))
print("*******************")
print("False Positive Rates")
print("*******************")
print(FP / (FP + TN))
print("*******************")

y_valid = pd.read_csv('/home/student/MachineLearningTest/Masters_Final_Project/Y_testing-0.1.csv', index_col=0)
labels = y_valid['PAGE_NUMBER'].unique()
y_valid = np_utils.to_categorical(y_valid['PAGE_NUMBER'].to_numpy())
y_true = y_valid.argmax(axis=1)
y_test = label_binarize(y_true, classes= labels)
y_pred = label_binarize(pre_cla.argmax(axis=1), classes=labels)
auc_keras = roc_auc_score(y_test, y_pred, multi_class='ovo')
print("AUC Score")
print("*******************")
print(auc_keras)

print("Plot the Micro Average of ROC of all classes")
print("*******************")

FPR, TPR, threshold_keras = roc_curve(y_test.ravel(), y_pred.ravel())

plt.figure()
plt.plot(FPR, TPR, label='Micro average ROC curve')

all_fpr = []
for i in range(0,100):
    try:
        all_fpr += [FPR[i],]
        FPR[i]
    except Exception:
        continue
all_fpr = np.unique(all_fpr)
mean_tpr = np.zeros_like(all_fpr)
for i in range(0,100):
    try:
        mean_tpr += interp(all_fpr, FPR[i], TPR[i])
    except Exception:
        continue
 
auc_curve = auc(FPR, TPR)

print("Plot the Macro Average of ROC of all classes")
print("*******************")

plt.figure()
plt.plot(FPR, TPR, label='Macro Average Curve')
plt.show()


gc.collect()







