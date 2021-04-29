import pandas as pd
import keras
import tensorflow as tf
import numpy as np
from keras.utils import np_utils
from TripletFingerprinting import TripletFingerPrintingNeuralNetwork
import random
import matplotlib.pyplot as plt
import sklearn.linear_model
import glob
import os
import gc
from zlib import crc32
import multiprocessing as mp

from sklearn.metrics import multilabel_confusion_matrix
from sklearn.model_selection import train_test_split
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

rootdir = '/home/student/MachineLearningTest/Masters_Final_Project/TestDirectory/AWS_zip/youtube/original_proc/'
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


final_training = pd.concat(training,axis=0,ignore_index=True).head(2500000)
print(final_training)

plt.scatter(final_training[['PACKET_SIZE']], final_training['PAGE_NUMBER'])
plt.show()
#plt.scatter(final_training[['TIME']], final_training['PAGE_NUMBER'])
#plt.show()
#plt.scatter(final_training[['SRC']], final_training['PAGE_NUMBER'])
#plt.show()
#plt.scatter(final_training[['DEST']], final_training['PAGE_NUMBER'])
#plt.show()
#print(final_training)

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

print(X)
print(Y)


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
y_train = np_utils.to_categorical(y_train.astype(int).to_numpy())
y_test = np_utils.to_categorical(y_test.astype(int).to_numpy())
print(y_train)
print(y_test)



INPUT_SHAPE = (2002,1)
new_triplet_set = X_train
new_test_set =  X_test

y_training = y_train.argmax(axis=1)

print(y_training)

classid_to_ids = {}
id_to_classid = {}
for i in range(len(y_training)):
    page_number = y_training[i]
    if page_number not in classid_to_ids:
        classid_to_ids[page_number] = [i,]
    else:
        classid_to_ids[page_number].append(i)

id_to_classid = {v: c for c, traces in classid_to_ids.items() for v in traces}


#testing
#classid_to_ids = {k: [path_to_id[path] for path in v] for k, v in new_triplet_set.items()}
#Triplet fingerprinting steps
def build_pos_pairs_for_id(classid): # classid --> e.g. 0
    try:
        traces = classid_to_ids[classid]
        pos_pairs = [(traces[i], traces[j]) for i in range(len(traces)) for j in range(i+1, len(traces))]
        return pos_pairs
    except Exception:
        print("Exception")


def build_positive_pairs(class_id_range):
    # class_id_range = range(0, num_classes)
    listX1 = []
    listX2 = []
    for class_id in pd.unique(y_training):
        pos = build_pos_pairs_for_id(class_id)
        # -- pos [(1, 9), (0, 9), (3, 9), (4, 8), (1, 4),...] --> (anchor example, positive example)
        for pair in pos:
            listX1 += [pair[0]] # identity
            listX2 += [pair[1]] # positive example
    perm = np.random.permutation(len(listX1))
    return np.array(listX1)[perm], np.array(listX2)[perm]


def build_pos_pairs_for_id_testing(classid): # classid --> e.g. 0
    try:
        traces = classid_to_ids[classid]
        pos_pairs = [(traces[i], traces[j]) for i in range(len(traces)) for j in range(i+1, len(traces))]
        return pos_pairs
    except Exception:
        print("Exception")


def build_positive_pairs_testing(class_id_range):
    # class_id_range = range(0, num_classes)
    listX1 = []
    listX2 = []
    for class_id in pd.unique(y_train):
        pos = build_pos_pairs_for_id_testing(class_id)
        # -- pos [(1, 9), (0, 9), (3, 9), (4, 8), (1, 4),...] --> (anchor example, positive example)
        for pair in pos:
            listX1 += [pair[0]] # identity
            listX2 += [pair[1]] # positive example
    perm = np.random.permutation(len(listX1))
    return np.array(listX1)[perm], np.array(listX2)[perm]

#Build similarity pair
def build_similarities(convolutional_model, packet_sizes):
    embs = convolutional_model.predict(packet_sizes)
    embs = embs / np.linalg.norm(embs, axis=-1, keepdims=True)
    return np.dot(embs, embs.T)



#Build negative pair
def build_negatives(anchor_indexes, positive_indexes, similarities, negative_indexes, num_retries=50):
    if similarities is None:
        return random.sample(negative_indexes,len(anchor_indexes))
    final_neg = []
    for (anchor_indexes, positive_indexes) in zip(anchor_indexes, positive_indexes):
        anchor_class = id_to_classid[anchor_indexes]
        possible_ids = list(set(negative_indexes) & set(np.where((similarities[anchor_indexes] + alpha_value) > similarities[anchor_indexes, positive_indexes])[0]))
        appended = False
        for iteration in range(num_retries):
            if len(possible_ids) == 0:
                break 
            idx_neg = random.choice(possible_ids)
            if id_to_classid[idx_neg] != anchor_class:
                final_neg.append(idx_neg)
                appended = True
                break
        if not appended:
            final_neg.append(random.choice(negative_indexes))
    return final_neg




def identity_loss(y_true, y_pred):
    neg_y_true = 1 - y_true
    neg_y_pred = 1 - y_pred
    false_positive = keras.backend.sum(neg_y_true * y_pred)
    true_negative = keras.backend.sum(neg_y_true * neg_y_pred)
    specificity  = true_negative / (true_negative + false_positive + keras.backend.epsilon())
    return  specificity

#Cosine Loss
def cosine_triplet_loss(X):
    positive_sim, negative_sim = X
    return keras.backend.mean(keras.backend.maximum(0.0, negative_sim - positive_sim + float(0.1))) 


convolutional_neural_network = TripletFingerPrintingNeuralNetwork(input=(2002,1), emb_vector_size=64)

anchor = keras.layers.Input((2002, 1), name='anchor')
positive = keras.layers.Input((2002, 1), name='positive')
negative = keras.layers.Input((2002, 1), name='negative')
a = convolutional_neural_network(anchor)
p = convolutional_neural_network(positive)
n = convolutional_neural_network(negative)
pos_sim = keras.layers.Dot(axes=-1, normalize=True)([a,p])
neg_sim = keras.layers.Dot(axes=-1, normalize=True)([a,n])
loss = keras.layers.Lambda(cosine_triplet_loss, output_shape=(1,))([pos_sim,neg_sim])

model_triplet = keras.models.Model(inputs=[anchor, positive, negative],outputs=loss)
opt = keras.optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)

model_triplet.compile(loss=identity_loss, optimizer=opt)

minimum=min(pd.unique(y_train.argmax(axis=1)))
maximum=max(pd.unique(y_train.argmax(axis=1)))
anchor_train, positive_train = build_positive_pairs(range(minimum, maximum + 1))
all_traces_train_idx = list(set(anchor_train) | set(positive_train))

print(new_triplet_set.shape)
print(anchor_train.shape)
print(positive_train.shape)
print(len(all_traces_train_idx))

#Triplet generator
class TripletGenerator():
    def __init__(self, anchor_train, positive_train, batch_size, packet_sizes, negative_traces_index, convolutional_model):
        self.batch_size = batch_size
        self.packet_sizes = packet_sizes
        self.anchor_train = anchor_train
        self.traces = packet_sizes
        self.positive_train = positive_train
        self.current_training_index = 0
        self.number_of_samples = anchor_train.shape[0]
        self.negative_traces_index = negative_traces_index
        self.all_anchors = list(set(anchor_train))
        self.mapping_pos = {v: k for k, v in enumerate(self.all_anchors)}
        self.mapping_neg = {k: v for k, v in enumerate(self.negative_traces_index)}
        if convolutional_model:
            self.similarities = build_similarities(convolutional_model, self.packet_sizes)
        else:
            self.similarities = None

    def next_train(self):
        while 1:
            self.current_training_index += self.batch_size
            if self.current_training_index >= self.number_of_samples:
                self.current_training_index = 0
            anchor_traces = self.anchor_train[self.current_training_index:self.current_training_index + self.batch_size]
            positive_traces = self.positive_train[self.current_training_index:self.current_training_index + self.batch_size]
            negative_traces = build_negatives(anchor_traces, positive_traces, self.similarities, self.negative_traces_index)
            yield ([self.traces[anchor_traces], self.traces[positive_traces], self.traces[negative_traces]], np.zeros(shape=(anchor_traces.shape[0])))


description = 'Triplet_Model'
training_csv_log = keras.callbacks.CSVLogger('log/Train_Log_%s.csv'%description, append=True, separator=';')
gen_hard = TripletGenerator(anchor_train, positive_train, 30, new_triplet_set, all_traces_train_idx, None)
epochs = 10
history = model_triplet.fit(gen_hard.next_train(), steps_per_epoch=30, epochs=1, verbose=1)
epochs_count = []
loss = []
for i in range(epochs - 1):
    history = model_triplet.fit(gen_hard.next_train(), steps_per_epoch=30, epochs=1, verbose=1)
    epochs_count += [i,]
    loss += [history.history['loss'],]
    gen_hard = TripletGenerator(anchor_train, positive_train, 30, new_triplet_set, all_traces_train_idx, convolutional_neural_network)

#summary of history for loss   
plt.plot(loss)
plt.plot(epochs_count)
plt.title('model loss')
plt.ylabel('loss')
plt.ylabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


#testing
new_triplet_set_testing = X_test
new_test_set = X_test
#y_training = pd.concat([y_train, y_valid])
y_testing = y_test.argmax(axis=1)

classid_to_ids_testing = {}
id_to_classids_testing = {}
for i in range(len(y_testing)):
    page_number = y_testing[i]
    if page_number not in classid_to_ids_testing:
        classid_to_ids_testing[page_number] = [i,]
    else:
        classid_to_ids_testing[page_number].append(i)

id_to_classids_testing = {v: c for c, traces in classid_to_ids_testing.items() for v in traces}

def build_pos_pairs_for_id_testing(classid): # classid --> e.g. 0
    try:
        traces = classid_to_ids_testing[classid]
        pos_pairs = [(traces[i], traces[j]) for i in range(len(traces)) for j in range(i+1, len(traces))]
        return pos_pairs
    except Exception:
        print("Exception")


def build_positive_pairs_testing(class_id_range):
    # class_id_range = range(0, num_classes)
    listX1 = []
    listX2 = []
    for class_id in pd.unique(y_testing):
        pos = build_pos_pairs_for_id_testing(class_id)
        # -- pos [(1, 9), (0, 9), (3, 9), (4, 8), (1, 4),...] --> (anchor example, positive example)
        for pair in pos:
            listX1 += [pair[0]] # identity
            listX2 += [pair[1]] # positive example
    perm = np.random.permutation(len(listX1))
    return np.array(listX1)[perm], np.array(listX2)[perm]

#Build similarity pair
def build_similarities(convolutional_model, packet_sizes):
    embs = convolutional_model.predict(packet_sizes)
    embs = embs / np.linalg.norm(embs, axis=-1, keepdims=True)
    return np.dot(embs, embs.T)



#Build negative pair
def build_negatives(anchor_indexes, positive_indexes, similarities, negative_indexes, num_retries=50):
    if similarities is None:
        return random.sample(negative_indexes,len(anchor_indexes))
    final_neg = []
    for (anchor_indexes, positive_indexes) in zip(anchor_indexes, positive_indexes):
        anchor_class = id_to_classid[anchor_indexes]
        possible_ids = list(set(negative_indexes) & set(np.where((similarities[anchor_indexes] + alpha_value) > similarities[anchor_indexes, positive_indexes])[0]))
        appended = False
        for iteration in range(num_retries):
            if len(possible_ids) == 0:
                break 
            idx_neg = random.choice(possible_ids)
            if id_to_classid[idx_neg] != anchor_class:
                final_neg.append(idx_neg)
                appended = True
                break
        if not appended:
            final_neg.append(random.choice(negative_indexes))
    return final_neg

class TripletGenerator():
    def __init__(self, anchor_train, positive_train, batch_size, packet_sizes, negative_traces_index, convolutional_model):
        self.batch_size = batch_size
        self.packet_sizes = packet_sizes
        self.anchor_train = anchor_train
        self.traces = packet_sizes
        self.positive_train = positive_train
        self.current_training_index = 0
        self.number_of_samples = anchor_train.shape[0]
        self.negative_traces_index = negative_traces_index
        self.all_anchors = list(set(anchor_train))
        self.mapping_pos = {v: k for k, v in enumerate(self.all_anchors)}
        self.mapping_neg = {k: v for k, v in enumerate(self.negative_traces_index)}
        if convolutional_model:
            self.similarities = build_similarities(convolutional_model, self.packet_sizes)
        else:
            self.similarities = None

    def next_train(self):
        while 1:
            self.current_training_index += self.batch_size
            if self.current_training_index >= self.number_of_samples:
                self.current_training_index = 0
            anchor_traces = self.anchor_train[self.current_training_index:self.current_training_index + self.batch_size]
            positive_traces = self.positive_train[self.current_training_index:self.current_training_index + self.batch_size]
            negative_traces = build_negatives(anchor_traces, positive_traces, self.similarities, self.negative_traces_index)
            yield ([self.traces[anchor_traces], self.traces[positive_traces], self.traces[negative_traces]], np.zeros(shape=(anchor_traces.shape[0])))



minimum_test=min(pd.unique(y_test.argmax(axis=1)))
maximum_test=max(pd.unique(y_test.argmax(axis=1)))
anchor_test, positive_test = build_positive_pairs_testing(range(minimum_test, maximum_test + 1))
all_traces_test_idx = list(set(anchor_test) | set(positive_test))

print(new_triplet_set_testing.shape)
print(anchor_test.shape)
print(positive_test.shape)
print(len(all_traces_test_idx))
print(y_test.shape)

gen_hard = TripletGenerator(anchor_test, positive_test, 30, new_triplet_set_testing, all_traces_test_idx, convolutional_neural_network)
pre_cla = model_triplet.predict(gen_hard.next_train(), verbose=1, steps=128)

print(pre_cla)
print(y_test.argmax(axis=1))








