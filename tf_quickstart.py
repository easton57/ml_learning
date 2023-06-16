import tensorflow as tf

# Load the dataset, convert data from ints to floats
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Build the model
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])

# Predictions
predictions = model(x_train[:1]).numpy()

# Convert to probabilities
tf.nn.softmax(predictions).numpy()

# Define Loss Function
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# Untrained output
print(loss_fn(y_train[:1], predictions).numpy())

# Optimize and compile model
model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5)

# Evaluate the model
model.evaluate(x_test,  y_test, verbose=2)
