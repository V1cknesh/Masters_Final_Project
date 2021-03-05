import keras

def TripletFingerPrintingNeuralNetwork(input=None, emb_vector_size=None):
	input_data = keras.layers.Input(shape=input)
	kernel_size = 8
	conv_stride_size = 1
	pool_stride_size = 4
	pool_size = 8

	model = keras.layers.Conv1D(filters=32, kernel_size=kernel_size, strides=conv_stride_size, padding='same', name='block1_conv1')(input_data)
	model = keras.layers.ELU(alpha=1.0, name='block1_adv_act1')(model)
	model = keras.layers.Conv1D(filters=32, kernel_size=kernel_size, strides=conv_stride_size, padding='same', name='block1_conv2')(model)
	model = keras.layers.ELU(alpha=1.0, name='block1_adv_act2')(model)
	model = keras.layers.MaxPooling1D(pool_size=pool_size, strides=pool_stride_size, padding='same', name='block1_pool')(model)
	model = keras.layers.core.Dropout(0.1, name='block1_dropout')(model)

	model = keras.layers.Conv1D(filters=64, kernel_size=kernel_size, strides=conv_stride_size, padding='same', name='block2_conv1')(model)
	model = keras.layers.core.Activation('relu', name='block2_act1')(model)
	model = keras.layers.Conv1D(filters=64, kernel_size=kernel_size, strides=conv_stride_size, padding='same', name='block2_conv2')(model)
	model = keras.layers.core.Activation('relu', name='block2_act2')(model)
	model = keras.layers.MaxPooling1D(pool_size=pool_size, strides=pool_stride_size, padding='same', name='block2_pool')(model)
	model = keras.layers.core.Dropout(0.1, name='block2_dropout')(model)

	model = keras.layers.Conv1D(filters=128, kernel_size=kernel_size, strides=conv_stride_size, padding='same', name='block3_conv1')(model)
	model = keras.layers.core.Activation('relu', name='block3_act1')(model)
	model = keras.layers.Conv1D(filters=128, kernel_size=kernel_size, strides=conv_stride_size, padding='same', name='block3_conv2')(model)
	model = keras.layers.core.Activation('relu', name='block3_act2')(model)
	model = keras.layers.MaxPooling1D(pool_size=pool_size, strides=pool_stride_size, padding='same', name='block3_pool')(model)
	model = keras.layers.core.Dropout(0.1, name='block3_dropout')(model)

	model = keras.layers.Conv1D(filters=256, kernel_size=kernel_size, strides=conv_stride_size, padding='same', name='block4_conv1')(model)
	model = keras.layers.core.Activation('relu', name='block4_act1')(model)
	model = keras.layers.Conv1D(filters=256, kernel_size=kernel_size, strides=conv_stride_size, padding='same', name='block4_conv2')(model)
	model = keras.layers.core.Activation('relu', name='block4_act2')(model)
	model = keras.layers.MaxPooling1D(pool_size=pool_size, strides=pool_stride_size, padding='same', name='block4_pool')(model)

	output = keras.layers.core.Flatten()(model)
	dense_layer = keras.layers.core.Dense(emb_vector_size, name='FeaturesVec')(output)
	final_model = keras.models.Model(inputs=input_data, outputs=dense_layer)
	return final_model
