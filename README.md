# DeepLPnet: Automated Prediction of Lattice Parameters from X-ray Powder Diffraction Patterns.

-----------------------------------------------------------------------------------------------------------------------------------------------------------------
This repository contains the trained models from the paper "Autonomous Prediction of Lattice Parameters from X-ray Powder Diffraction Patterns". The models here can be used to predict on unknown synchrotron PXRD patterns for which the crystal system is known. For model training, please contact ICSD and CSD for access to their databases which were utilized in our analysis. 

Key Results: Prediction of lattice parameters with ~ 10% mean percentage error (MPE) for each crystal system. Analysis of realistic modifications which cause ML methods to underperform (baseline noise, broadening, intensity modulation, impurity phases, zero-shifting). Quantification of search space reduction around true lattice parameters (~ 100 - 1000 fold reduction). Strong prediction for experimental synchrotron data from Beamline 2-1 at SSRL. 

**Authors: Sathya Chitturi, Daniel Ratner, Richard Walroth, Vivek Thampy, Evan Reed, Mike Dunne, Chris Tassone and Kevin Stone.**
**Affiliations: Stanford University / SLAC National Accelerator Laboratory**

Paper Link: https://doi.org/10.1107/S1600576721010840
---

This directory contains the following files: 

**src** 

* *model.py*: contains the regression model (LPregressor) in the analysis. 
* *helper_functions.py*: helper functions for ML predictions, training, augmentation and analysis
* *helper_functions_topas.py*: helper functions for automatic Topas script generation for Lp-Search. 
* *data_generation.py*: code for the dataloader used for modifications (augmentation) during training.

**data** 

* The SSRL Beamline 2-1 dataset is available in the XY_data_labels directory.

**examples** 

* *predictions.ipynb*: is an example jupyter notebook to show how to deploy the ML models on data collected from Beamline 2-1 at SSRL. 
* *topasScriptGeneration.ipynb*: is an example jupyter notebook to show how to create a corresponding topas file for Lp-Search (https://journals.iucr.org/j/issues/2017/05/00/to5164/) based on the ML predictions for SSRL data. 

**models_ICSD_CSD** 

* folder contains all trained models for the no-modification, baseline noise, broadening, multiphase experiments and all-modifications. These models were trained on ICSD and CSD as described in the paper. 

---

**Installation** 

1) Make a new local folder and clone the repository

* git clone https://github.com/src47/DeepLPnet.git

2) Upgrade pip if needed

* pip3 install --upgrade pip

3) Make a virtual environment to install packages and activate. Here we use venv but other environments such as conda will also work. 

* python3 -m venv env 
* source env/bin/activate

4) Install relevant packages

* pip3 install -r requirements.txt

Note: the skued package used for baselining is slightly unstable and sometimes does not install correctly. If this is case, one option is to comment the import statement in helper_functions.py and remove scikit-ued from requirements.txt. In this scenario, the rest of the code base works, however the baselining option will not be available. 

**Please direct any questions or comments to chitturi@stanford.edu and khstone@slac.stanford.edu.** 

