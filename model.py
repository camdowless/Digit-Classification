import tensorflow as tf

data = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = data.load_data()

# normalize data
x_train = x_train / 255.0
x_test = x_test / 255.0
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

# Add color channel. turns the shape from (60000, 28, 28) to (60000, 28, 28, 1)
x_train = x_train.reshape(
    x_train.shape[0], x_train.shape[1], x_train.shape[2], 1
)
x_test = x_test.reshape(
    x_test.shape[0], x_test.shape[1], x_test.shape[2], 1
)

model = tf.keras.Sequential([
    # Convolutional layer with 32 filters, 3x3 kernel
    tf.keras.layers.Conv2D(
        32, (3, 3), activation="relu", input_shape=(28, 28, 1)
    ),
    # Max pooling layer
    tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
    # Flatten
    tf.keras.layers.Flatten(),

    # Hidden layer
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(.5),

    # Output layer for 10 digits
    tf.keras.layers.Dense(10, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(x_train, y_train, epochs=10)
model.evaluate(x_test, y_test, verbose=2)
model.save("model.h5")
