import tensorflow as  tf
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation

num_house = 100
np.random.seed(42)
house_size = np.random.randint(low=1000, high=3500, size=num_house)
np.random.seed(42)
house_price = house_size * 100.0 + np.random.randint(low=20000, high=70000, size=num_house)


tf.compat.v1.disable_eager_execution()

sess = tf.compat.v1.Session()

hello = tf.constant("Test constant")
print(sess.run(hello))
a = tf.constant(20)
b = tf.constant(22)
print('a + b = {0}'.format(sess.run(a+b)))

plt.plot(house_size, house_price, "bx")
plt.ylabel("Price")
plt.xlabel("Size")
plt.show()

def normalize(array):
	return (array - array.mean()) / array.std()

num_train_samples = math.floor(num_house * 0.7)
train_house_size = np.asarray(house_size[:num_train_samples])
train_price = np.asarray(house_price[:num_train_samples:])
train_house_size_norm = normalize(test_house_size)
train_price_norm = normalize(test_house_size)

tf_house_size = tf.placeholder("float", name="house_size")
tf_price = tf.placeholder("float", name="price") 

tf_size_factor = tf.Variable(np.random.randn(), name="size_factor")
tf_price_offset = tf.Variable(np.random.randn(), name="price_offset")

tf_price_pred = tf.add(tf.multiply(tf_size_factor, tf_house_size), tf_price_offset)
tf_cost = tf.reduce_sum(tf.pow(tf_price_pred-tf_price, 2))/(2*num_train_samples)

learning_rate = 0.1
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(tf_cost)
init = tf.global_variables_initializer() 
