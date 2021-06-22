import tensorflow as tf
from helper_functions import * 

class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, data_directory, list_IDs, labels, X_random=np.zeros((100,9000)), batch_size=64, dimx=(9000), dimy=(3), 
                 shuffle=True,augmentation=False, shift_offset=False, intensity_shift=False, linear_comb=False, gaussian_noise=False, gaussian_broaden=False,
                shift=15, percent_scale=0.30, num_examples=3, impurity_scale=0.10,noise_level=0.002, probability=1.0, sigma=1.0):
        'Initialization'
        self.augmentation = augmentation      # boolean whether to perform augmentation 
        self.data_directory = data_directory  # path to data directory 
        self.dimx = dimx                      # number of features 
        self.dimy = dimy                      # number of outputs 
        self.batch_size = batch_size          
        self.labels = labels                  
        self.list_IDs = list_IDs              # shuffled list of IDs 
        self.shuffle = shuffle
        self.on_epoch_end()
        
        # Augmentation types 
        self.shift_offset = shift_offset
        self.intensity_shift = intensity_shift
        self.linear_comb = linear_comb
        self.gaussian_noise = gaussian_noise
        self.gaussian_broaden = gaussian_broaden
        
        # Augmentation amount 
        self.shift = shift                     # Amount of horizontal displacement 
        self.percent_scale = percent_scale     # Intensity scale factor 
        self.num_examples = num_examples       # Number of linear combination 
        self.impurity_scale = impurity_scale   # Scale factor for impurities
        self.X_random = X_random               # Matrix for linear combinations; default is a zero-matrix 
        self.noise_level = noise_level         # Baseline noise threshold 
        self.probability = probability         # Probability that pattern has baseline noise 
        self.sigma = sigma                     # Broadening factor 
        
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size] # Generate indexes of the batch
        list_IDs_temp = [self.list_IDs[k] for k in indexes] # Find list of IDs
        X, y = self.__data_generation(list_IDs_temp) # Generate data
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, self.dimx, 1))
        y = np.empty((self.batch_size, self.dimy))
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            file = os.path.join(self.data_directory, "data_" + str(ID+1) + ".npy")
            x = np.load(file)
            x = x[0:self.dimx]
            # Store sample
            X[i,] = np.reshape(x, (x.shape[0], 1))
            y[i,] = self.labels[ID]
        
        # Flag to set whether train with augmentation
        if self.augmentation:
            X = augment(X, self.X_random, shift_offset=self.shift_offset, intensity_shift=self.intensity_shift, linear_comb=self.linear_comb, 
                gaussian_noise=self.gaussian_noise,gaussian_broaden=self.gaussian_broaden, shift=self.shift, percent_scale=self.percent_scale,  
                num_examples=self.num_examples, impurity_scale=self.impurity_scale,noise_level=self.noise_level, probability=self.probability, sigma=self.sigma)
        return X, y


        
 
