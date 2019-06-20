import tensorflow as tf
print(tf.__version__)

# load mnist
mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

# display sample data
import matplotlib.pyplot as plt
print(training_labels[42])
print(training_images[42])
plt.imshow(training_images[42])
plt.show()

# normalize data
training_images = training_images / 255.0
test_images = test_images / 255.0

# define neural network
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(128, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

# fit data and labels
model.compile(optimizer = tf.train.AdamOptimizer(),
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(training_images, training_labels, epochs=5)

# test data
model.evaluate(test_images, test_labels)
