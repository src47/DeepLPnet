from matplotlib import pyplot as plt
import numpy as np 
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import gaussian_filter1d
import tensorflow as tf
import os 
from scipy import signal 
import scipy
from skued import baseline_dt

def make_prediction(X, model, crystal_system):
    # Use this function for final prediction to ensure correct symmetry for lattice parameters 
    y_pred = model.predict(X)
    if ((crystal_system == "hexagonal") or (crystal_system == "cubic") or (crystal_system == "tetragonal") or (crystal_system == "trigonal")):
        enforce_symmetry(y_pred,crystal_system)
    return y_pred
  
def enforce_symmetry(prediction_array, crystal_system):

    # This function correctly enforces a a c for trigonal, tetragonal, hexagonal and cubic crystals;
    # No return is needed since arrays are passed in by reference in python; i.e. prediction_array is overwritten
    if ((crystal_system == "hexagonal") or (crystal_system == "tetragonal") or (crystal_system == "trigonal")):
        for prediction in prediction_array:
            if abs(prediction[0]-prediction[1]) < abs(prediction[0]-prediction[2]):
                prediction[0] = (prediction[0]+prediction[1])/2
                prediction[1] = prediction[0]
            else:
                prediction[1] = (prediction[1] + prediction[2]) / 2
                prediction[2] = prediction[1]
    if (crystal_system == "cubic"):
        for prediction in prediction_array:
            prediction[0] = np.mean(prediction)
            prediction[1] = prediction[0]
            prediction[2] = prediction[0]

def normalize01(X):
    # Normalize data to 0 1 range 
    Xnew = []
    for i in range(len(X)):
        norm = ((X[i] - np.min(X[i]))/((np.max(X[i]) - np.min(X[i])) + 0.000001))
        Xnew.append(norm)
    Xnew = np.array(Xnew)
    return Xnew

# Augmentations 
def augment(X, X_random, shift_offset=True, intensity_shift=True, linear_comb=True, gaussian_noise=True,gaussian_broaden=True, shift=15, percent_scale=0.30, num_examples=4, impurity_scale=0.10,noise_level=0.005, probability=1.0, sigma=1.0):

    if len(X.shape) == 3:  # Need to reduce dimensions in order for other calculations to work
        X = np.reshape(X, (X.shape[0], X.shape[1]))
    if shift_offset:
        X = shift_spectra(X, shift)
    if intensity_shift:
        X = intensity_modulation(X, percent_scale)
    if linear_comb:
        X = linear_combination(X, X_random, num_examples, impurity_scale)
    if gaussian_noise:
        X = gaussian_noise_baseline(X, noise_level, probability)
    if gaussian_broaden:
        X = gaussian_broaden_data(X)
    X = np.reshape(X, (X.shape[0], X.shape[1],1))
    return X

def shift_spectra(X, shift=10):

    # Random shift between -shift and shift; Based on code for shifting numpy arrays: https://stackoverflow.com/questions/30399534/shift-elements-in-a-numpy-array
    shift = np.random.randint(shift) - int(shift/2)
    augmented = np.empty_like(X)
    if shift > 0:
        augmented[:, :shift] = 0
        augmented[:, shift:] = X[:, :-shift]
    elif shift < 0:
        augmented[:, shift:] = 0
        augmented[:, :shift] = X[:, -shift:]
    else:
        augmented[:, :] = X
    return augmented

def intensity_modulation(X, percent_scale=0.20):
    
    # Random intensity modulation 
    X += X*np.repeat(np.random.uniform(-percent_scale, percent_scale, size=(X.shape[0], 100)), X.shape[1]/100, axis=1)
    return normalize01(X)

def linear_combination(X, X_random, num_examples=3, impurity_scale=0.10):
    # Random number between 1 and num_examples for linear combination adding
    if num_examples != 0:
        num_combinations = np.random.randint(num_examples) + 1
    else:
        num_combinations = 1
    batch_size = X.shape[0]
    X_random = X_random[0:batch_size]
    for i in range(num_combinations):
        X += np.random.uniform(0.05, impurity_scale, size=(batch_size, 1)) * (np.random.permutation(X_random)[0:batch_size])
        X = normalize01(X)
    return X

def gaussian_noise_baseline(X, noise_level=0.02, probability=1.0):
    
    # Add gaussian noise to the baseline
    if np.random.rand() < probability:
        X += np.random.uniform(0,noise_level, size=(X.shape[0], 1))*np.random.normal(noise_level, 1, size=(X.shape[0],X.shape[1]))
        #X += abs(np.random.normal(0, noise_level/3, size=(X.shape[0],X.shape[1]))) # for changing baseline noise experiments 
    return abs(X)

def gaussian_broaden_data(X):
    
    # Add gaussian broaden to the data 
    sigma = np.random.uniform(1, 5)
    return normalize01(gaussian_filter1d(X, sigma, axis=1))

def mean_absolute_percentage_error(y_true, y_pred):
    
    # Calculate MPE 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / (y_true + 0.0000001))) * 100

def find_peaks_wrapper(x,widths=np.arange(1,20)):
    peakind = signal.find_peaks_cwt(x, widths)        
    out = np.zeros(len(x))
    out[peakind] = x[peakind]
    out = out/np.max(out)
    return out

def angle2q(two_theta, lbda=1.541838):
    return (4*np.pi*np.sin((two_theta/2)*np.pi/180))/lbda

def interpolate_data(q, intensities, lbda=1.541838,ml_input_size=9000):
    f = scipy.interpolate.interp1d(q, intensities, bounds_error=False, fill_value=intensities[0])
    q_sim = angle2q(np.linspace(0, 90, ml_input_size), lbda)
    intensities_interpolated = f(q_sim)
    return intensities_interpolated

def add_axis_broaden(intensity_interpolated,sigma=3):
    x = scipy.ndimage.gaussian_filter1d(intensity_interpolated, sigma=5)
    x = x/np.max(x)
    x = np.expand_dims(x, axis=1)
    x = np.expand_dims(x, axis=0)
    return x 

def processExptData(Xdata, measured_wavelength=0.7293, showPlots=True, baseline=False):
    q = angle2q(Xdata[0], lbda=measured_wavelength)
    intensity = Xdata[1]
    
    if baseline: # If data has a non-zero baseline, we can use autobaselining tools: https://scikit-ued.readthedocs.io/en/master/
        intensity = intensity - baseline_dt(intensity, wavelet = 'qshift3', level = 9, max_iter = 1000)
    
    intensity = intensity/np.max(intensity)
    intensity[intensity < 0.001] = 0
    
    intensity_interpolated = interpolate_data(q, intensity, lbda=1.54056,ml_input_size=9000) # Interpolate to 9000 range in corresponding q 
    intensity_interpolated = intensity_interpolated/np.max(intensity_interpolated) # normalize to 0,1

    if showPlots:
        plt.plot(np.linspace(0,90,9000),intensity_interpolated)
        plt.show()
        
    return intensity_interpolated

def predictExptDataPipeline(Xdata, y_true, crystal_system, measured_wavelength=0.7293, model=None, baseline=False):

    if model == None:
        # Default model takes all augmentations 
        model = tf.keras.models.load_model("models_ICSD_CSD/" + crystal_system +  "_all")
        
    intensity_interpolated = processExptData(Xdata, measured_wavelength=measured_wavelength, showPlots=True, baseline=baseline)
    y_pred = make_prediction(np.expand_dims(np.expand_dims(intensity_interpolated,axis=1),axis=0), model, crystal_system)
    
    print(" ")
    print("True LPs from Refined data: ", y_true)
    print(" ")
    print("Predicted LPs using ML: ", y_pred)    
    print("----------------------------------------------------------------------------------------------------------------")
    print("----------------------------------------------------------------------------------------------------------------")
    print(" ")
    
    return y_pred

