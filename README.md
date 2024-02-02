# SPCESM2-ML ensemble - stochastic and deterministic ML ensembles in the SPCESM2 framework 



Author: Gunnar Behrens - <gunnar.behrens@dlr.de> 

This repository is to some extent based on the CBRAIN repository of Stephan Rasp (https://github.com/raspstephan/CBRAIN-CAM): 

Main Repository Author: Stephan Rasp - <raspstephan@gmail.com> - https://raspstephan.github.io

Thank you for checking out our SPCESM2-ML ensembles repository, dedicated to building stochastic and deterministic ensembles for learning convective processes in SPCESM2. This is a working repository, which means that the most current commit might not always be the most functional or documented. For a quick-start guide to use the SPCAM or SPCESM2 data you may have a look on the quick-start notebook of Stephan Rasp that can be found here: https://github.com/raspstephan/CBRAIN-CAM/blob/master/quickstart.ipynb

The current release of SPCESM2-ML ensemble on zenodo can be found here (to be updated): 

<![![DOI](https://zenodo.org/badge/227609774.svg)](https://zenodo.org/badge/latestdoi/227609774)>




If you are looking for the exact version of the code that corresponds to the PNAS paper, check out this release: https://github.com/raspstephan/CBRAIN-CAM/releases/tag/PNAS_final [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1402384.svg)](https://doi.org/10.5281/zenodo.1402384)

For a sample of the SPCESM data, prepocessed data and initilization files of CEM2 used, click here: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10598576.svg)](https://doi.org/10.5281/zenodo.10598576)


The modified earth system model code is available at https://github.com/SciPritchardLab/CESM2-ML-coupler . 


### Papers using this repository

> G.Behrens, T. Beucler, F. Iglesias-Suarez, S. Yu, P. Gentine, M. Pritchard, M. Schwabe and V. Eyring, 2024.
> Improving Atmospheric Processes in Earth System Models with Deep Learning Ensembles
> and Stochastic Parameterizations






## Repository description

The main components of the repository are:

- `cbrain`: Contains the cbrain module with all code to preprocess the raw data, run the neural network experiments and analyze the data based on Stephan Rasp repository (https://github.com/raspstephan/CBRAIN-CAM).
- `models`: Contains all files need to build the ensemble parameterization. All weight, training, history files of individual machine learning algorithms (Artificial Neural Networks (ANNs), Variational Encoder Decoders, an ANN with dropout) can be found  in the subfolder `offline_models`. The subfolder `online_models`  contains all necessary model files to conduct online experiments with CESM2.
- `environments`: Contains the .yml files of the conda environments used for this repository 
- `pp_config`: Contains configuration files and shell scripts to preprocess the eart system model data to be used as neural network inputs
- `CRPS_analysis`: Contains files to evaluate the ensembles based on the Continous Rank Probabilty Score (CRPS)
- `deterministic_analysis`: Contains files to evaluate the ensembles based on coefficient of determination (R2) and mean absolute error (MAE)
- `latent_perturbation_tuning`: Contains files that are necessary to adjust the magnitude of the latent space perturbation of Variational Encoder Decoders.
- `online_evaluation`: Contains the files that were used to evaluate the skill of the hybrid simulations with CESM2. 
- `online_run_scripts`: Contains example run scripts of CESM2 with machine learning parameterizations. These were created in collaboration with Sungduk Yu. 
- `uncertainty_quantification`: Contains files to evaluate the quality of the ensembles with repect to uncertainty quantification (UQ). It includes code that is based on the published repository for UQ of Cathy Haynes and Ryan Lagerqvist. 
- `preprocessing_real_geography.py`: Is the related python code to preprocess raw SPCESM2 data into .nc files that can be used with the cbrain repository of Stephan Rasp.  
- `List_of_Figures.txt`: Contains a description where to find the python code to reproduce the figures of the SPCESM2-ML ensemble paper
