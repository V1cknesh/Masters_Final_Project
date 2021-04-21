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
from DeepFingerprinting4 import DeepFingerprintingNeuralNetwork
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
import matplotlib.pyplot as plt
import pandas as pd

threshold="0.05"
threshold3=0.05
social_media_site="fb"
alpha_value=0.05

rootdir = '/home/student/MachineLearningTest/Masters_Final_Project/TestDirectory/AWS_zip/fb/original_proc/'
SOURCE_IPADDRESS = ['172.31.40', '172.31.47', '172.31.36', '172.31.46', '172.31.33']

charset = 'abcdefghijklmnopqrstuvwxyz1234567890'

def ipaddressconverter(ipaddress): 
    return 1 if '.'.join(ipaddress.split('.')[0:-1]) in SOURCE_IPADDRESS  else 0;

def encoding(bin_string):
    bin_string = str(bin_string)
    binary_data = (bin_string).encode("utf-8")
    c = Compressor()
    c.use_gzip()
    ascii_string = c.compress(binary_data)
    return ascii_string

training = []
testing = []
final_training = pd.read_csv('/home/student/MachineLearningTest/Masters_Final_Project/TestDirectory/AWS_zip/fb/original_proc/25/497.csv')
count = 0
for subdir, dirs, files in os.walk(rootdir):
    for subdirs, dirs, files in os.walk(subdir):
        for file in files:
            count += 1
            if count < 30:
                page_number = subdirs.split("/")[-1]
                filename = subdirs + "/" + file
                df = pd.read_csv(filename, names=['TIME', 'SRC', 'DEST', 'PACKET_SIZE'])
                length_of_data = len(df.index)
                lower_quartile = int(0.15 * length_of_data)
                upper_quartile = int(0.85 * length_of_data)
                df.drop_duplicates(inplace=True)
                df = df[0: len(df.index): 1]
                df['SRC'] = pd.to_numeric(df['SRC'].transform(func=lambda x: ipaddressconverter(x)), errors='coerce').astype('float32')
                df['DEST'] = pd.to_numeric(df['DEST'].transform(func=lambda x: ipaddressconverter(x)), errors='coerce').astype('float32')
                df['PAGE_NUMBER'] = page_number
                training.append(df)
            else:
                count = 0
                break


final_training = pd.concat(training,axis=0,ignore_index=True).head(5000000)
print(final_training)

plt.scatter(final_training[['PACKET_SIZE']], final_training['PAGE_NUMBER'])
plt.show()


length_of_source_address = 0
F = []
k = 1
B = []
SOURCE_ADDRESS2 = []
DEST_ADDRESS2 = []
cumulative_packet_list = []
time = 0
initial_time = final_training['TIME'].iloc[0]
group_packet_size = 0
page_number = -1
cumulative_packets2 = 0
count = 0
for index, row in final_training.iterrows():
    count += 1
    if (count <= 2000):
        #group_packet_size += row['PACKET_SIZE']
        cumulative_packet_list += [row['PACKET_SIZE'],] 
        time += row['TIME']
        page_number = row['PAGE_NUMBER']
        SOURCE_ADDRESS2 += [row['SRC'],]
        DEST_ADDRESS2 += [row['DEST'],]
        initial_time = row['TIME']
    elif (count > 2000):
        try:
            if len(cumulative_packet_list) < 2000:
                cumulative_packet_list += [0] * (2000 - len(cumulative_packet_list))
            elif len(cumulative_packet_list) > 2000:
                cumulative_packet_list = cumulative_packet_list[0:2000]
            if len(SOURCE_ADDRESS2) < 2000:
                SOURCE_ADDRESS2 += [0] * (2000 - len(SOURCE_ADDRESS2))
            elif len(SOURCE_ADDRESS2) > 2000:
                SOURCE_ADDRESS2 = SOURCE_ADDRESS2[0:2000]
            if len(DEST_ADDRESS2) < 2000:
                DEST_ADDRESS2 += [0] * (2000 - len(DEST_ADDRESS2))
            elif len(DEST_ADDRESS2) > 2000:
                DEST_ADDRESS2 = DEST_ADDRESS2[0:2000]
            test = [time,group_packet_size] + cumulative_packet_list + [page_number,]
            F.append(test)
            group_packet_size = 0
            SOURCE_ADDRESS2 = []
            DEST_ADDRESS2 = []
            cumulative_packet_list = []
            time = 0
            initial_time = row['TIME']
            count = 0
        except Exception:
            continue
        

final_training = pd.DataFrame(F)
final_training.drop_duplicates(inplace=True)
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
INPUT_SHAPE = (2002,1)
NUMBER_OF_PAGES=101
y_train = np_utils.to_categorical(y_train.astype(int).to_numpy())
y_test = np_utils.to_categorical(y_test.astype(int).to_numpy())


#DeepFingerprinting Steps

model = DeepFingerprintingNeuralNetwork.neuralnetwork(input=INPUT_SHAPE, N=NUMBER_OF_PAGES)
model.summary()
history = model.fit(X_train, y_train, batch_size=1000,shuffle=True, epochs=60, verbose=1, validation_data=(X_test, y_test))


print(history.history)
plt.plot(history.history['categorical_accuracy'])
plt.plot(history.history['val_categorical_accuracy'])
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

y_test = y_test
labels = y_test['PAGE_NUMBER'].unique()
print(labels)
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







