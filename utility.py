import pandas as pd
import keras
import tensorflow as tf
import numpy as np
from DeepFingerprinting import DeepFingerprintingNeuralNetwork
from TripletFingerprinting import TripletFingerprintingNeuralNetwork
import random
import os


random.seed(0)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
NUMBER_OF_PAGES=100
INPUT_SHAPE = (5000,1)

#Training and Testing datasets

training_dataset = pd.read_csv('/home/student/MachineLearningTest/Masters_Final_Project/training_data_49-0_1.csv', names=['INDEX', 'TIME', 'SRC', 'PACKET_SIZE', 'CUMULATIVE_PACKET_SIZE', 'PAGE_NUMBER', 'TOTAL_PACKET_SIZE'], index_col=None, skiprows=[1])
testing_dataset = pd.read_csv('/home/student/MachineLearningTest/Masters_Final_Project/testing_data_49-0_1.csv', names=['INDEX', 'TIME', 'SRC', 'PACKET_SIZE', 'CUMULATIVE_PACKET_SIZE', 'PAGE_NUMBER', 'TOTAL_PACKET_SIZE'], index_col=None, skiprows=[1])


X_train = pd.to_numeric(training_dataset['TOTAL_PACKET_SIZE'], errors='coerce').astype('float32')
X_valid = pd.to_numeric(testing_dataset['TOTAL_PACKET_SIZE'], errors='coerce').astype('float32')
y_train = training_dataset['PAGE_NUMBER']
y_valid = testing_dataset['PAGE_NUMBER']

# we need a [Length x 1] x n shape as input to the DFNet (Tensorflow)
#X_train = X_train[:, :,np.newaxis]
#X_valid = X_valid[:, :,np.newaxis]

print(X_train.shape[0], 'training samples')
print(X_valid.shape[0], 'validation samples')

# convert class vectors to binary class matrices
#y_train = tf.keras.utils.np_utils.to_categorical(y_train, NB_CLASSES)
#y_valid = tf.keras.utils.np_utils.to_categorical(y_valid, NB_CLASSES)

model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28,28]))
model.add(keras.layers.Dense(300, activation="relu"))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(10, activation="softmax"))
print("done")

#DeepFingerprinting Steps
model = DeepFingerprintingNeuralNetwork.neuralnetwork(input=INPUT_SHAPE, N=NUMBER_OF_PAGES)
final_model = model.fit(X_train, y_train, batch_size=128, epochs=30, verbose=1, validation_data=(X_valid, y_valid))
print(final_model.best_params_)
print(final_model.best_score_)
model = final_model.best_estimator_.model



#Triplet fingerprinting steps
def build_positive_pairs(page_number_range):
    identity_list = []
    positive_list = []
    traces = np.arange(NUMBER_OF_PAGES)
    for page_number in page_number_range:
        positive_pairs = random.shuffle([(traces[i], traces[j]) for i in range(len(traces)) for j in range(i+1, len(traces))])
        for pair in positive_pairs:
            identity_list += [pair[0]] # identity
            positive_list += [pair[1]] # positive example
    random_permutation = np.random.permutation(len(identity_list))
    return np.array(identity_list)[random_permutation], np.array(positive_list)[random_permutation]

anchor_train, positive_train = build_positive_pairs(range(0, num_classes))
all_traces_train_idx = list(set(anchor_train) | set(positive_train))


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



#Run the convolutional neeural network
convolutional_neural_network = TripletFingerprintingNeuralNetwork(input_shape=(5000,1), emb_size=64)


#Combining the triplet loss into a model
def identity_loss(y_true, y_pred):
    return keras.backend.K.mean(y_pred - 0 * y_true)

#Cosine Loss
def cosine_triplet_loss(X):
    positive_sim, negative_sim = X
    return keras.backend.K.mean(keras.backend.K.maximum(0.0, negative_sim - positive_sim + float(0.1)))

anchor = convolutional_neural_network(INPUT_SHAPE)
positive = convolutional_neural_network(INPUT_SHAPE)
negative = convolutional_neural_network(INPUT_SHAPE)
pos_sim = keras.layers.Dot(axes=-1, normalize=True)([anchor,positive])
neg_sim = keras.layers.Dot(axes=-1, normalize=True)([anchor,negative])

#Customized loss function
loss = keras.layers.Lambda(cosine_triplet_loss, output_shape=(1,))([pos_sim,neg_sim])

#Creating the triplet model
model_triplet = keras.models.Model(inputs=[anchor, positive, negative], outputs=loss)
opt = keras.optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model_triplet.compile(loss=identity_loss, optimizer=opt)


#Triplet generator
class TripletGenerator():
    def __init__(self, anchor_train, positive_train, batch_size, packet_sizes, negative_traces_index, convolutional_model):
        self.batch_size = batch_size
        self.packet_sizes = packet_sizes
        self.anchor_train = anchor_train
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
            if self.current_training_index >= self.num_samples:
                self.current_training_index = 0
            anchor_traces = self.anchor_train[self.current_training_index:self.current_training_index + self.batch_size]
            positive_traces = self.positive_train[self.current_training_index:self.current_training_index + self.batch_size]
            negative_traces = build_negatives(anchor_traces, positive_traces, self.similarities, self.negative_traces_index)
            yield ([self.traces[anchor_traces], self.traces[positive_traces], self.traces[negative_traces]], np.zeros(shape=(anchor_traces.shape[0])))



#Log the training output
csv_log = keras.callbacks.CSVLogger('log/Train_Log_%s.csv'%description, append=True, separator=';')
# At first epoch we don't generate hard triplets
gen_hard = TripletGenerator(anchor_train, positive_train, 128, all_traces, all_traces_train_idx, None)
epochs = 30
for epoch in range(epochs):
    model_triplet.fit_generator(generator=gen_hard.next_train(), steps_per_epoch=anchor_train.shape[0] // 128, epochs=1, verbose=1, callbacks=[csv_log])
    gen_hard = TripletGenerator(anchor_train, positive_train, 128, all_traces, all_traces_train_idx, convolutional_neural_network)




