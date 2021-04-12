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
from DeepFingerprinting3 import DeepFingerprintingNeuralNetwork
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from kerastuner.tuners import RandomSearch
import kerastuner as kt
from sklearn.preprocessing import label_binarize
from keras.preprocessing.text import hashing_trick
import hashlib
from sklearn.model_selection import train_test_split
from compress import Compressor

threshold="0.05"
threshold3=0.05
social_media_site="fb"
alpha_value=0.05

rootdir = '/home/student/MachineLearningTest/Masters_Final_Project/dataset/fb/0.1_data'

SOURCE_IPADDRESS = ['172.31.40', '172.31.47', '172.31.36', '172.31.46', '172.31.33']

charset = 'abcdefghijklmnopqrstuvwxyz1234567890'

def ipaddressconverter(ipaddress): 
    return 1 if '.'.join(ipaddress.split('.')[0:-1]) in SOURCE_IPADDRESS  else -1;

def encoding(bin_string):
    bin_string = str(bin_string)
    binary_data = (bin_string).encode("utf-8")
    c = Compressor()
    c.use_gzip()
    ascii_string = c.compress(binary_data)
    return ascii_string

training = []
testing = []
count = 0
for subdir, dirs, files in os.walk(rootdir):
    if (subdir.split("_")[-1] == "data"):
        threshold=subdir.split("/")[-1].split("_")[0]
        for subdirs, dirs, files in os.walk(subdir):
            for file in files:
                filename = subdir + "/" + file
                if file.split("_")[0] == "training":
                    df = pd.read_csv(filename, index_col=0)[1:]
                    length_of_data = len(df.index)
                    lower_quartile = int(0.15 * length_of_data)
                    upper_quartile = int(0.85 * length_of_data)
                    df.drop_duplicates(inplace=True)
                    #df = df[0: len(df.index): 1]
                    df['PACKET_SIZE'] = pd.to_numeric(df['PACKET_SIZE'], errors='coerce').astype('float32')
                    df['TOTAL_PACKET_SIZE'] = pd.to_numeric(df['TOTAL_PACKET_SIZE'], errors='coerce').astype('float32')
                    df['CUMULATIVE_PACKET_SIZE'] = pd.to_numeric(df['CUMULATIVE_PACKET_SIZE'], errors='coerce').astype('float32')
                    df['TIME'] = pd.to_numeric(df['TIME'], errors='coerce').astype('float32')
                    df['SRC'] = pd.to_numeric(df['SRC'].transform(func=lambda x: ipaddressconverter(x)), errors='coerce').astype('float32')
                    df['PACKET_SIZE'] = df['PACKET_SIZE'].div(df['PACKET_SIZE'].sum(), axis=0)
                    #df['TOTAL_PACKET_SIZE'] = df['TOTAL_PACKET_SIZE'].div(df['TOTAL_PACKET_SIZE'].sum(), axis=0)
                    #df['CUMULATIVE_PACKET_SIZE'] = df['CUMULATIVE_PACKET_SIZE'].div(df['CUMULATIVE_PACKET_SIZE'].sum(), axis=0)
                    #df['TIME'] = df['TIME'].div(df['TIME'].sum(), axis=0)
                    df['SRC'] = df['SRC'].replace(-1, 0)
                    training.append(df)
                else:
                    tf = pd.read_csv(filename, index_col=0)[1:]
                    length_of_data = len(tf.index)
                    lower_quartile = int(0.15 * length_of_data)
                    upper_quartile = int(0.85 * length_of_data)
                    tf.drop_duplicates(inplace=True)
                    #tf = tf[0: len(tf.index): 1]
                    tf['PACKET_SIZE'] = pd.to_numeric(tf['PACKET_SIZE'], errors='coerce').astype('float32')
                    tf['TOTAL_PACKET_SIZE'] = pd.to_numeric(tf['TOTAL_PACKET_SIZE'], errors='coerce').astype('float32')
                    tf['CUMULATIVE_PACKET_SIZE'] = pd.to_numeric(tf['CUMULATIVE_PACKET_SIZE'], errors='coerce').astype('float32')
                    tf['TIME'] = pd.to_numeric(tf['TIME'], errors='coerce').astype('float32')
                    tf['SRC'] = pd.to_numeric(tf['SRC'].transform(func=lambda x: ipaddressconverter(x)), errors='coerce').astype('float32')
                    tf['PACKET_SIZE'] = tf['PACKET_SIZE'].div(tf['PACKET_SIZE'].sum(), axis=0)
                    #tf['TOTAL_PACKET_SIZE'] = tf['TOTAL_PACKET_SIZE'].div(tf['TOTAL_PACKET_SIZE'].sum(), axis=0)
                    #tf['CUMULATIVE_PACKET_SIZE'] = tf['CUMULATIVE_PACKET_SIZE'].div(tf['CUMULATIVE_PACKET_SIZE'].sum(), axis=0)
                    #tf['TIME'] = tf['TIME'].div(tf['TIME'].sum(), axis=0)
                    tf['SRC'] = tf['SRC'].replace(-1, 0)
                    training.append(tf)

            final_training = pd.concat(training,axis=0,ignore_index=True)[:-1]

            length_of_source_address = 0
            F = []
            k = 1
            B = []
            SOURCE_ADDRESS = ""
            SOURCE_ADDRESS2 = []
            cumulative_packet_list = []
            time = 0
            initial_time = final_training['TIME'].iloc[0]
            group_packet_size = 0
            page_number = -1
            cumulative_packets2 = 0
            for index, row in final_training.iterrows():
                if (row[0] - initial_time < 0.005):
                    group_packet_size += row[2]
                    cumulative_packet_list += [group_packet_size,]
                    time += row[0]
                    page_number = row[4]
                    SOURCE_ADDRESS += str(row[1])
                    SOURCE_ADDRESS2 += [row[1],]
                else:
                    try:
                        cumulative_packets = str(bin(int.from_bytes(encoding(cumulative_packets2), "little"))[2:])
                        if length_of_source_address == 0:
                            length_of_source_address = len(SOURCE_ADDRESS)
                        else:
                            if len(SOURCE_ADDRESS) < length_of_source_address:
                                continue
                            else:
                                SOURCE_ADDRESS = str(bin(int.from_bytes(encoding(str(SOURCE_ADDRESS.replace(".", ""))), "little"))[2:])
                                if len(cumulative_packet_list) < 10:
                                    cumulative_packet_list += [0] * (10 - len(cumulative_packet_list))
                                elif len(cumulative_packet_list) > 10:
                                    cumulative_packet_list = cumulative_packet_list[0:10]
                                if len(SOURCE_ADDRESS) < 146:
                                    SOURCE_ADDRESS += [0] * (146 - len(SOURCE_ADDRESS))
                                elif len(SOURCE_ADDRESS) > 146:
                                    SOURCE_ADDRESS = SOURCE_ADDRESS[0:146]
                                if len(SOURCE_ADDRESS2) < 148:
                                    SOURCE_ADDRESS2 += [0] * (148 - len(SOURCE_ADDRESS2))
                                elif len(SOURCE_ADDRESS2) > 148:
                                    SOURCE_ADDRESS2 = SOURCE_ADDRESS2[0:148]
                                #SOURCE_ADDRESS2 = str(SOURCE_ADDRESS).replace("", ".").split(".")
                                #SA = np.frombuffer(SOURCE_ADDRESS, dtype=float, sep='')
                                #print(SA)
                                test = [time,group_packet_size] + cumulative_packet_list + SOURCE_ADDRESS2 + [page_number,]
                                F.append(test)
                                group_packet_size = 0
                                SOURCE_ADDRESS = ""
                                cumulative_packet_list = []
                                time = 0
                                k += 1
                    except Exception:
                        continue
                    

            final_training = pd.DataFrame(F)
            print(final_training)

X = final_training[final_training.columns[:-1]]
Y = final_training[final_training.columns[-1]]


random.seed(0)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state =1)
X_train = X_train.to_numpy().astype("float32")
X_test = X_test.to_numpy().astype("float32")


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

X_train = X_train[:, :,np.newaxis]
X_test = X_test[:, :,np.newaxis]
INPUT_SHAPE = (160,1)
NUMBER_OF_PAGES=101
y_train = np_utils.to_categorical(y_train.astype(int).to_numpy())
y_test = np_utils.to_categorical(y_test.astype(int).to_numpy())


#DeepFingerprinting Steps

model = DeepFingerprintingNeuralNetwork.neuralnetwork(input=INPUT_SHAPE, N=NUMBER_OF_PAGES)
model.summary()
history = model.fit(X_train, y_train, batch_size=100,shuffle=True, epochs=30, verbose=1, validation_data=(X_test, y_test))
print(history.history)
plt.plot(history.history['categorical_accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.ylabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#summary of history for loss   
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.ylabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

y_pred = model.predict(X_test, verbose=1, batch_size=100)
print(y_test)
print(y_pred)

cm = multilabel_confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))

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

y_test = pd.read_csv('/home/student/MachineLearningTest/Masters_Final_Project/Y_testing-0.1.csv', index_col=0)
labels = y_test['PAGE_NUMBER'].unique()
y_test = np_utils.to_categorical(y_test['PAGE_NUMBER'].to_numpy())
y_true = y_test.argmax(axis=1)

print(y_true)
print(y_pred.argmax(axis=1))
y_test = label_binarize(y_true, classes= labels)
y_pred = label_binarize(y_pred.argmax(axis=1), classes=labels)
auc_keras = roc_auc_score(y_test.argmax(axis=1), y_pred.argmax(axis=1), multi_class='ovo')
print("AUC Score")
print("*******************")
print(auc_keras)







