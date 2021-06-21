# LPnet: Deep Learning Estimation of Crystal Lattice Parameters from X-Ray Powder Diffraction Patterns

-----------------------------------------------------------------------------------------------------------------------------------------------------------------
This repository contains the trained models from the paper "Deep Learning Estimation of Crystal Lattice Parameters from X-Ray Powder Diffraction Patterns". Please contact ICSD and CSD for access to their databases which were utilized in our analysis. 

Key Results: Prediction of lattice parameters with ~ 10% mean percentage error (MPE) for each crystal system. Analysis of realistic modifications which cause ML methods to underperform (baseline noise, broadening, intensity modulation, impurity phases, zero-shifting). Quantification of search space reduction around true lattice parameters (~ 100 - 1000 fold reduction). 

**Authors: Sathya Chitturi, Daniel Ratner, Richard Walroth, Vivek Thampy, Evan Reed, Mike Dunne, Chris Tassone and Kevin Stone.**
**Affiliations: Stanford University / SLAC National Accelerator Laboratory**

**Tutorial Notebooks:** 

* *ssrl_predictions.ipynb*: is an example jupyter notebook to show how to deploy the ML models on data collected from Beamline 2-1 at SSRL. 

**Data:** 

* The SSRL Beamline 2-1 dataset is available in the XY_data_labels directory 

**SRC:** 

* *model.py*: contains the regression model (LPregressor) in the analysis. It also contains two other models which use inception blocks and skip connections which seem to work better. 

* *helper_functions.py*: various helper functions used in the analysis. 

* *data_generation.py*: code for the dataloader used for modifications (augmentation) during training.

**models_ICSD_CSD**: 

* folder contains all trained models for the no-modification, baseline noise, broadening and multiphase experiments. These models were trained on ICSD and CSD as described in the paper. 

**Installation:** 
* *requirements.txt*: All dependencies can be installed using pip install -r requirements.txt 

**Please direct any questions or comments to chitturi@stanford.edu or khstone@slac.stanford.edu!** 

