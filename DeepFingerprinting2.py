import keras


class DeepFingerprintingNeuralNetwork:
	@staticmethod
	def neuralnetwork(input, N):
		model = keras.Sequential()
		no_of_internal_layers = 5
		kernel = 4
		pool = 2
		conv_stride = 1
		pool_stride = 2
		optimization_function = keras.optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
		sgd = keras.optimizers.SGD(learning_rate=0.1, momentum=1e-6, nesterov=False)
		kernel_initialization = keras.initializers.glorot_uniform(seed=0)

		#Block layer 1
		model.add(keras.layers.Conv1D(filters=4, kernel_size=kernel, input_shape=input, strides=conv_stride,activation='relu', padding='same', name='block1_convolutional_layer1'))
		#model.add(keras.layers.BatchNormalization(axis=-1))
		#model.add(keras.layers.ELU(alpha=1.0, name='block1_activation_layer1'))
		model.add(keras.layers.Conv1D(filters=32, kernel_size=kernel, strides=conv_stride, padding='same',activation='relu', name='block1_convolutional_layer2'))
		#model.add(keras.layers.BatchNormalization(axis=-1))
		#model.add(keras.layers.ELU(alpha=1.0, name='block1_activation_layer2'))
		#model.add(keras.layers.MaxPooling1D(pool_size=pool, strides=pool_stride, padding='same', name='block1_pooling_layer'))
		model.add(keras.layers.core.Dropout(0.5, name='block1_dropout_layer'))

		#Block layers 2,3,4 connected layers
		for layer in range(1, no_of_internal_layers + 1):
			name1 = "block"+str(layer)+"_convolutional_layer" + str(layer + 2)
			name2 = "block"+str(layer)+"_convolutional_layer" + str(layer + 3)
			model.add(keras.layers.ZeroPadding1D(padding=1, name="zero_padding"+str(layer)+"_layer"))
			model.add(keras.layers.Conv1D(filters=32 * layer + 1, kernel_size=kernel, strides=conv_stride, activation='relu',  padding='same', name=name1)) #Block Convolution layer 1
			#model.add(keras.layers.BatchNormalization()) #Batch normalization
			model.add(keras.layers.core.Activation('relu', name="block"+str(layer)+"_activation_layer" + str(layer + 2))) #Block RELU Activation layer 1
			model.add(keras.layers.Conv1D(filters=32 * layer + 1, kernel_size=kernel, strides=conv_stride, activation='relu',  padding='same', name=name2)) #Block Convolution layer 1
			#model.add(keras.layers.BatchNormalization()) #Batch normalization
			model.add(keras.layers.core.Activation('relu', name="block1"+str(layer)+"_activation_layer" + str(layer + 3))) #Block RELU Activation layer 2
			#model.add(keras.layers.MaxPooling1D(pool_size=pool, strides=pool_stride, padding='same')) #Max Pooling
			model.add(keras.layers.core.Dropout(0.5, name="block2"+str(layer)+"_dropout")) #Block Dropout layer

		#Out3put prediction
		model.add(keras.layers.core.Flatten(name='flatten'))
		model.add(keras.layers.core.Dense(N, kernel_initializer=kernel_initialization, name='fully_connected_layer3')) #Where N is the number of pages we are testing for
		model.add(keras.layers.core.Activation('softmax', name="softmax"))

		model.compile(loss='categorical_crossentropy', optimizer='adam' , metrics=['accuracy', keras.metrics.SensitivityAtSpecificity(0.5)])
		return model
