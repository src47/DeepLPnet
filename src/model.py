import tensorflow as tf
from helper_functions import *

class CNNBlock(tf.keras.Model):
    def __init__(self, filters, kernel_size, pool_size):
        super(CNNBlock, self).__init__()
        
        self.conv1D_0 = tf.keras.layers.Conv1D(filters, kernel_size, activation="relu", padding="same")
        self.conv1D_1 = tf.keras.layers.Conv1D(filters, kernel_size, activation="relu", padding="same")
        self.max_pool = tf.keras.layers.MaxPool1D(pool_size=pool_size)
        
    def call(self, inputs):
        
        x = self.conv1D_0(inputs)
        x = self.conv1D_1(x)
        x = self.max_pool(x)
        
        return x
    
# This is the model that is used in the paper 
class LPregressor(tf.keras.Model):

    def __init__(self, num_outputs):
        super(LPregressor, self).__init__()
        
        self.initial_pool = tf.keras.layers.MaxPool1D(pool_size=3) # Pool down to 3000 features 
        # Blocks of CNN + CNN + MaxPool 
        self.block_a = CNNBlock(5, 3, 2)
        self.block_b = CNNBlock(10, 3, 2)
        self.block_c = CNNBlock(15, 5, 3)
        self.block_d = CNNBlock(20, 5, 2)
        self.block_e = CNNBlock(30, 5, 5)
        # Flatten 
        self.flatten = tf.keras.layers.Flatten()
        # FC + FC +FC + out 
        self.fc1 = tf.keras.layers.Dense(80, activation="relu")
        self.fc2 = tf.keras.layers.Dense(50, activation="relu")
        self.fc3 = tf.keras.layers.Dense(10, activation="relu")
        self.out = tf.keras.layers.Dense(num_outputs, activation='linear')

    def call(self, inputs):
        
        x = self.initial_pool(inputs)
        x = self.block_a(inputs)
        x = self.block_b(x)
        x = self.block_c(x)
        x = self.block_d(x)
        x = self.block_e(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.out(x)
        return x
