import keras

class DeepFingerprintingNeuralNetwork:
	@staticmethod
	def neuralnetwork(input, N):
		model = keras.Sequential()
		no_of_internal_layers = 5
		kernel = 1
		pool = 2
		conv_stride = 1
		pool_stride = 2
		optimization_function = keras.optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
		sgd = keras.optimizers.SGD(learning_rate=0.1, momentum=1e-6, nesterov=False)
		kernel_initialization = keras.initializers.glorot_uniform(seed=0)

		#model.add(keras.layers.Conv1D(filters=1, input_shape=input, kernel_size=kernel, activation='relu',  name='block1_convolutional_layer1'))
		
		#Output prediction
		model.add(keras.layers.core.Flatten(name='flatten'))
		model.add(keras.layers.core.Dense(N, kernel_initializer=kernel_initialization, name='fully_connected_layer6')) #Where N is the number of pages we are testing for
		model.add(keras.layers.core.Activation('softmax', name="softmax"))

		model.compile(loss='categorical_crossentropy', optimizer='adam' , metrics=['categorical_accuracy', keras.metrics.SensitivityAtSpecificity(0.5)])
		return model