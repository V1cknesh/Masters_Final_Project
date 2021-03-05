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

rootdir = '/home/student/MachineLearningTest/Masters_Final_Project/dataset/youtube/0.09_data'


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

alpha_value = float(0.1)

X_train = pd.read_csv('/home/student/MachineLearningTest/Masters_Final_Project/X_training-0.09.csv', index_col=0)
y_train = pd.read_csv('/home/student/MachineLearningTest/Masters_Final_Project/Y_training-0.09.csv', index_col=0)
X_valid = pd.read_csv('/home/student/MachineLearningTest/Masters_Final_Project/X_testing-0.09.csv', index_col=0)
y_valid = pd.read_csv('/home/student/MachineLearningTest/Masters_Final_Project/Y_testing-0.09.csv', index_col=0)


NUMBER_OF_PAGES=len(pd.unique(y_train['PAGE_NUMBER']))
#new_triplet_set = pd.concat([X_train, X_valid])
new_triplet_set = X_train.to_numpy()[:, :,np.newaxis]
new_test_set = X_train.to_numpy()[:, :,np.newaxis]
#y_training = pd.concat([y_train, y_valid])
y_training = y_train

classid_to_ids = {}
id_to_classid = {}
for i in range(len(y_training)):
    page_number = y_training['PAGE_NUMBER'].iloc[i]
    if page_number not in classid_to_ids:
        classid_to_ids[page_number] = [i,]
    else:
        classid_to_ids[page_number].append(i)

id_to_classid = {v: c for c, traces in classid_to_ids.items() for v in traces}



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
    for class_id in pd.unique(y_train['PAGE_NUMBER']):
        pos = build_pos_pairs_for_id(class_id)
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


convolutional_neural_network = TripletFingerPrintingNeuralNetwork(input=(5,1), emb_vector_size=64)

anchor = keras.layers.Input((5, 1), name='anchor')
positive = keras.layers.Input((5, 1), name='positive')
negative = keras.layers.Input((5, 1), name='negative')
a = convolutional_neural_network(anchor)
p = convolutional_neural_network(positive)
n = convolutional_neural_network(negative)
pos_sim = keras.layers.Dot(axes=-1, normalize=True)([a,p])
neg_sim = keras.layers.Dot(axes=-1, normalize=True)([a,n])
loss = keras.layers.Lambda(cosine_triplet_loss, output_shape=(1,))([pos_sim,neg_sim])

model_triplet = keras.models.Model(inputs=[anchor, positive, negative],outputs=loss)
#print(model_triplet.summary())

opt = keras.optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)

model_triplet.compile(loss=identity_loss, optimizer=opt)

minimum=min(pd.unique(y_train['PAGE_NUMBER']))
maximum=max(pd.unique(y_train['PAGE_NUMBER']))
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
prediction_csv_log = keras.callbacks.CSVLogger('log/Prediction_Log_%s.csv'%description, append=True, separator=';')
gen_hard = TripletGenerator(anchor_train, positive_train, 30, new_triplet_set, all_traces_train_idx, convolutional_neural_network)
epochs = 1
for epoch in range(epochs): 
    model_triplet.fit_generator(generator=gen_hard.next_train(), steps_per_epoch= 30, epochs=1, verbose=1, callbacks=[training_csv_log])
    gen_hard = TripletGenerator(anchor_train, positive_train, 30, new_triplet_set, all_traces_train_idx, convolutional_neural_network)






