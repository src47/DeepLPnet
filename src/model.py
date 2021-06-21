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

class RESBlock(tf.keras.Model):
    def __init__(self, filters, kernel_size, pool_size):
        super(RESBlock, self).__init__()
        
        self.conv1D_0 = tf.keras.layers.Conv1D(filters, kernel_size, activation="relu", padding="same")
        self.conv1D_1 = tf.keras.layers.Conv1D(filters, kernel_size, activation="relu", padding="same")
        self.conv1D_2 = tf.keras.layers.Conv1D(filters, kernel_size, activation="linear", padding="same")
        self.conv1_shortcut = tf.keras.layers.Conv1D(filters, 1, activation="linear", padding="same")
        self.max_pool = tf.keras.layers.MaxPool1D(pool_size=pool_size)
        self.add = tf.keras.layers.Add()
        self.activation = tf.keras.layers.Activation('relu')
        
    def call(self, inputs):
        
        short_cut = self.conv1_shortcut(inputs)
        
        x = self.conv1D_0(inputs)
        x = self.conv1D_1(x)
        x = self.conv1D_2(x)
        x = self.add([x, short_cut])
        x = self.activation(x)
        x = self.max_pool(x)
        
        return x

class INCEPBlock(tf.keras.Model):
    def __init__(self, filters, kernel_size_1, kernel_size_2, kernel_size_3, kernel_size_4, pool_size):
        super(INCEPBlock, self).__init__()
        
        self.conv1D_0 = tf.keras.layers.Conv1D(filters, kernel_size_1, activation="relu", padding="same")
        self.conv1D_1 = tf.keras.layers.Conv1D(filters, kernel_size_2, activation="relu", padding="same")
        self.conv1D_2 = tf.keras.layers.Conv1D(filters, kernel_size_3, activation="relu", padding="same")
        self.conv1D_3 = tf.keras.layers.Conv1D(filters, kernel_size_4, activation="relu", padding="same")
        self.max_pool = tf.keras.layers.MaxPool1D(pool_size=pool_size)
        
    def call(self, inputs):
                
        x1 = self.conv1D_0(inputs)
        x2 = self.conv1D_1(inputs)
        x3 = self.conv1D_2(inputs)
        x4 = self.conv1D_3(inputs)
        x = tf.keras.layers.Concatenate()([x1, x2, x3, x4])
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
        
        #x = self.initial_pool(inputs)
        x = self.block_a(inputs)
        #x = self.block_a(x)
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
    
# This model performs slightly better than the reported model for some crystal systems 
class LPregressorSkipConnections(tf.keras.Model):

    def __init__(self, num_outputs):
        super(LPregressorSkipConnections, self).__init__()
        
        self.initial_pool = tf.keras.layers.MaxPool1D(pool_size=3) # Pool down to 3000 features 
        # Blocks of CNN + CNN + MaxPool 
        self.block_a = RESBlock(5, 3, 2)
        self.block_b = RESBlock(10, 3, 2)
        self.block_c = RESBlock(15, 5, 3)
        self.block_d = RESBlock(20, 5, 2)
        self.block_e = RESBlock(30, 5, 5)
        # Flatten 
        self.flatten = tf.keras.layers.Flatten()
        # FC + FC +FC + out 
        self.fc1 = tf.keras.layers.Dense(80, activation="relu")
        self.fc2 = tf.keras.layers.Dense(50, activation="relu")
        self.fc3 = tf.keras.layers.Dense(10, activation="relu")
        self.out = tf.keras.layers.Dense(num_outputs, activation='linear')

    def call(self, inputs):
        
        x = self.initial_pool(inputs)        
        x = self.block_a(x)
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
    
# This model performs slightly better than the reported model for some crystal systems 
class LPregressorIncep(tf.keras.Model):

    def __init__(self, num_outputs):
        super(LPregressorIncep, self).__init__()
        
        self.initial_pool = tf.keras.layers.MaxPool1D(pool_size=3) # Pool down to 3000 features 
        # Blocks of CNN + CNN + MaxPool 
        self.block_a = INCEPBlock(2, 3, 5, 7, 9, 2)
        self.block_b = INCEPBlock(3, 3, 5, 7, 9, 2)
        self.block_c = RESBlock(15, 5, 3)
        self.block_d = RESBlock(20, 5, 2)
        self.block_e = RESBlock(30, 5, 5)
        # Flatten 
        self.flatten = tf.keras.layers.Flatten()
        # FC + FC +FC + out 
        self.fc1 = tf.keras.layers.Dense(80, activation="relu")
        self.fc2 = tf.keras.layers.Dense(50, activation="relu")
        self.fc3 = tf.keras.layers.Dense(10, activation="relu")
        self.out = tf.keras.layers.Dense(num_outputs, activation='linear')

    def call(self, inputs):
        
        x = self.initial_pool(inputs)        
        x = self.block_a(x)
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