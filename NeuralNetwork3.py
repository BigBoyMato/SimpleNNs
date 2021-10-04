import os

# GPU USAGE NOTIFICATIONS OFF 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.layers import Dense

# returns monthly budget balance  # PREDICT FUNCTION
def monthly_balance(outcome, income, credit):
    return (income - outcome) - (credit*1.1/12)

# Learning Selection
#  monthly spent -USD || monthly income - USD || credit total - USD
inp_ls = [[1200, 1700, 10000], [2000, 2800, 15000], [3000, 4500, 10000], 
          [2000, 4200, 12000], [3100, 3400, 3000], [3000, 3200, 4000], 
          [1900, 2500, 13000], [2000, 2200, 6000], [1500, 2100, 5000]]
inp = np.array(inp_ls)

# average month-balance to pay
out_ls = [-417, -575, 583, 1100, 25, -167, -592, -350, 142]
out = np.array(out_ls)

# creating NN abstract model
model = keras.Sequential()

#creating NN layers
model.add(Dense(units=3, input_shape=(3,), activation='linear'))
model.add(Dense(units=1, input_shape=(3,), activation='linear'))

# compiling NN 
model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(0.1))

# training NN
history = model.fit(inp, out, epochs=500, verbose=0)

# Checking error loss
plt.plot(history.history['loss'])
plt.grid(True)
plt.show()

# Check predictions
print(model.predict(np.array([[5000, 6000, 20000]]))) # -833
print(monthly_balance(5000, 6000, 20000))
print('\n')
print(model.predict(np.array([[1500, 2000, 10000]]))) # -417
print(monthly_balance(1500, 2000, 10000))
print('\n')
print(model.predict(np.array([[2400, 3500, 1000]]))) # 1008
print(monthly_balance(2400, 3500, 1000))
print('\n')

# get set weights
print(model.get_weights())
