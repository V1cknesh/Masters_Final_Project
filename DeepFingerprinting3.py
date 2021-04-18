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
		optimization_function2 = keras.optimizers.Adam(lr=0.0001, decay=1e-6)
		sgd = keras.optimizers.SGD(learning_rate=0.1, momentum=1e-6, nesterov=False)
		kernel_initialization = keras.initializers.glorot_uniform(seed=0)

		model.add(keras.layers.Dense(256, activation='relu'))
		model.add(keras.layers.core.Activation('softmax', name="softmax1"))
		model.add(keras.layers.Dense(128, activation='relu'))
		model.add(keras.layers.core.Activation('softmax', name="softmax2"))
		model.add(keras.layers.Dense(64, activation='relu'))
		model.add(keras.layers.core.Activation('softmax', name="softmax3"))
		model.add(keras.layers.core.Dense(N, kernel_initializer=kernel_initialization, name='fully_connected_layer3')) #Where N is the number of pages we are testing for
		model.add(keras.layers.core.Activation('softmax', name="softmax4"))

		model.add(keras.layers.core.Flatten(name='flatten'))
		model.add(keras.layers.core.Dense(N, kernel_initializer=kernel_initialization, name='fully_connected_layer4')) #Where N is the number of pages we are testing for
		model.add(keras.layers.core.Activation('softmax', name="softmax"))
		model.compile(loss='categorical_crossentropy', optimizer=optimization_function2 , metrics=[ keras.metrics.TopKCategoricalAccuracy(),keras.metrics.CategoricalAccuracy(), keras.metrics.SensitivityAtSpecificity(0.5)])
		return model
