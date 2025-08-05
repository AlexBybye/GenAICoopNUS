import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import LambdaCallback
def plot_graphs(process, metric, filename):
    plt.plot(process.history[metric])
    plt.title(filename.split('.')[0])
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.savefig(filename)
    plt.show()
xs = np.array([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0], dtype=float)
ys = np.array([-8.0, -5.0, -2.0, 1.0, 4.0, 7.0], dtype=float)
inputs = keras.Input(shape = (1,))
outputs = layers.Dense(1)(inputs)
model = keras.Model(inputs = inputs, outputs = outputs)
model.compile(optimizer='sgd', loss='mean_absolute_error')
history = model.fit(xs, ys, epochs=500)
plot_graphs(history, 'loss', '../First NN - Loss.png')
print(model.predict(np.array([12.0])))
print("Model weights: {}".format(model.get_weights()))
"""
inputs = keras.Input(shape = (1,))
outputs = layers.Dense(1)(inputs)

model = keras.Model(inputs = inputs, outputs = outputs)
model.compile(optimizer='sgd', loss='mean_absolute_error')

# specify the callback in fit()
model.fit(xs, ys, epochs=500, callbacks=weight_callback)

# we print out the weights stored in our dictionary
for epoch, weights in weights_dict.items():
    print("Value of m for epoch #",epoch+1)
    print(weights[0])
    print("Value of c for epoch #",epoch+1)
    print(weights[1])    
"""
# weights_dict = {}
# # define our custom callback
# # at the end of every epoch, we will update our dictionary with the weights of the model
# weight_callback = LambdaCallback(on_epoch_end=lambda epoch, logs: weights_dict.update({epoch:model.get_weights()}))