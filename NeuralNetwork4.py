import os

# GPU USAGE NOTIFICATIONS OFF 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist  # db with handwritten number images
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten

# x_train - training selection images
# y_train - vector of corresponding values for training selection
# x_test - test selection images
# y_test - vector of corresponding values for test selection
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# value normalization
x_train = x_train / 255
x_test = x_test / 255
y_train_cat = keras.utils.to_categorical(y_train, 10)
y_test_cat = keras.utils.to_categorical(y_test, 10)

# VIEW IMAGES FROM DB 
# ---------- ---------- ---------- #
plt.figure(figsize=(10, 5))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_train[i], cmap=plt.cm.binary)

plt.show()
# ---------- ---------- ---------- #


# BUILDING NN
# ---------- ---------- ---------- #

# 28x28=784 input neurons - equivalent to pixel image size
# 128 hidden layer neurons 
# 10 output layer neurons - equivalent to possible output (nums)
# hiddent neuron activate function - ReLu 
# ouput neuron activate function - softmax (probability of belonging to num class)

model = keras.Sequential([
    Flatten(input_shape=(28, 28, 1)), # input layer (transform image to pixel vector)
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

print(model.summary()) # check model 

model.compile(optimizer='adam',
loss='categorical_crossentropy',
metrics=['accuracy'])

# 32 pictrues as a batch size, 10 epochs, 20% validation 
model.fit(x_train, y_train_cat, batch_size=32, epochs=10,
validation_split=0.2)

model.evaluate(x_test, y_test_cat)
# ---------- ---------- ---------- #


# NN TESTING 
# ---------- ---------- ---------- #
n = 3
x = np.expand_dims(x_test[n], axis=0)
res = model.predict(x)
print( res )

plt.imshow(x_test[n], cmap=plt.cm.binary)
plt.show()

# Test all selection
pred = model.predict(x_test)
pred = np.argmax(pred, axis=1)

print(pred.shape)

print(pred[:20])
print(y_test[:20])

# mark invalid results
mask = pred == y_test
print(mask[:10])

x_false = x_test[~mask]
y_false = x_test[~mask]

print(x_false.shape)

# Show first 25 invalid results
plt.figure(figsize=(10,5))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_false[i], cmap=plt.cm.binary)

plt.show()
# ---------- ---------- ---------- #

