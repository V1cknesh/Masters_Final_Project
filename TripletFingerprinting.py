import keras

class TripletFingerprintingNeuralNetwork:
	@staticmethod
	def neuralnetwork(input, emb_vector_size):
		model = keras.Sequential()
		no_of_internal_layers = 3
		kernel = 8
		pool = 8
		conv_stride = 1
		pool_stride = 4
		optimization_function = Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
		kernel_initialization = keras.initializers.glorot_uniform(seed=0)
		input_data = keras.Input(shape=input)

		#Block layer 1
		model.add(keras.layers.Conv1D(filters=32, kernel_size=kernel, input_shape=input_data, strides=conv_stride, padding='same', name='block1_convolutional_layer1'))
		model.add(keras.layers.BatchNormalization(axis=-1))
		model.add(keras.layers.ELU(alpha=1.0, name='block1_activation_layer1'))
		model.add(keras.layers.Conv1D(filters=32, kernel_size=kernel, strides=conv_stride, padding='same', name='block1_convolutional_layer2'))
		model.add(keras.layers.BatchNormalization(axis=-1))
		model.add(keras.layers.ELU(alpha=1.0, name='block1_activation_layer1'))
		model.add(keras.layers.MaxPooling1D(pool_size=pool, strides=pool_stride, padding='same', name='block1_pooling_layer'))
		model.add(keras.layers.core.Dropout(0.1, name='block1_dropout_layer'))

		#Block layers 2,3,4 connected layers
		for layer in range(1, no_of_internal_layers + 1):
			model.add(keras.layers.Conv1D(filters=32 * layer + 1, kernel_size=kernel, strides=conv_stride, padding='same')) #Block Convolution layer 1
			model.add(keras.layers.BatchNormalization()) #Batch normalization
			model.add(keras.layers.core.Activation('relu')) #Block RELU Activation layer 1
			model.add(keras.layers.Conv1D(filters=32 * layer + 1, kernel_size=kernel, strides=conv_stride, padding='same')) #Block Convolution layer 1
			model.add(keras.layers.BatchNormalization()) #Batch normalization
			model.add(keras.layers.core.Activation('relu')) #Block RELU Activation layer 2
			model.add(keras.layers.MaxPooling1D(pool_size=pool, strides=pool_stride, padding='same')) #Max Pooling
			model.add(keras.layers.core.Dropout(0.1, name='block2_dropout')) #Block Dropout layer

		#Fully connected layer1
		model.add(keras.layers.core.Flatten(name='flatten'))
		model.add(keras.layers.core.Dense(512, kernel_initializer=kernel_initialization, name='fully_connected_layer1'))
		model.add(keras.layers.BatchNormalization())
		model.add(keras.layers.core.Activation('relu', name='fully_connected_activation1'))
		model.add(keras.layers.core.Dropout(0.7, name='fully_connected_dropout1'))

		#Fully connected layer1
		model.add(keras.layers.core.Dense(512, kernel_initializer=kernel_initialization, name='fully_connected_layer2'))
		model.add(keras.layers.BatchNormalization())
		model.add(keras.layers.core.Activation('relu', name='fully_connected_activation1'))
		model.add(keras.layers.core.Dropout(0.5, name='fully_connected_dropout2'))

		#Output prediction
		model.add(keras.layers.core.Flatten(name='flatten2'))
		model.add(keras.layers.core.Dense(emb_vector_size, kernel_initializer=kernel_initialization, name='fully_connected_layer3')) #Where emb_vector_size is the vector size of the embedded vector
		model.compile(loss="sparse_categorical_crossentropy", optimizer=keras.optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0), metrics=["accuracy"])
		return model

