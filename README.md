# SPCESM2-ML ensemble - stochastic and deterministic ML ensembles in the SPCESM2 framework 



Author: Gunnar Behrens - <gunnar.behrens@dlr.de> 

This repository uses parts of the code from the CBRAIN repository of Stephan Rasp (https://github.com/raspstephan/CBRAIN-CAM): 

Main Repository Author: Stephan Rasp - <raspstephan@gmail.com> - https://raspstephan.github.io

Thank you for checking out our SPCESM2-ML ensembles repository, dedicated to building stochastic and deterministic ensembles for learning convective processes in SPCESM2. A quick-start notebook to use the SPCAM or SPCESM2 data can be found here: https://github.com/raspstephan/CBRAIN-CAM/blob/master/quickstart.ipynb

The current release of SPCESM2-ML ensemble on zenodo can be found here (to be updated): 

<![![DOI](https://zenodo.org/badge/227609774.svg)](https://zenodo.org/badge/latestdoi/227609774)>





For a sample of the SPCESM data, prepocessed data and initilization files of CESM2 used, click here: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10598576.svg)](https://doi.org/10.5281/zenodo.10598576)


The modified earth system model code is available at https://github.com/SciPritchardLab/CESM2-ML-coupler. 


### The Paper using this repository

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


## Data:

1) A set of SPCESM2 data can be found on Zenodo: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10598576.svg)](https://doi.org/10.5281/zenodo.10598576)
   - It contains a subset of the preprocessed training data (first day of each month of 2013), validation data (first day of each month of 2014) and test data (first day of each of 2015) that is used in the SPCESM2-ML ensemble paper.
   - It includes a folder with SPCESM2 raw data consisting of .nc files from January 1st 2013.
   - It includes a folder with initilization files of SPCESM2 for January 2013 and Febrauary 2013, that can be used to run hybrid CESM2 simulations with the here developed ML ensembles.

2) The entire set of SPCESM2 raw data, preprocessed SPCESM2 data and more SPCESM2 initilization files for different months of 2013 is archived on Levante/DKRZ and available upon request.
   This allows to reproduce all figures and all results of the SPCESM2-ML ensemble paper. In this case an account on [DKRZ/Levante](https://docs.dkrz.de/) is needed.

## Figures:

A completed list of all figures of the above mentioned paper and the related path in the repository can be found [here](https://github.com/EyringMLClimateGroup/behrens24james_SPCESM2_ML_ensembles/blob/main/List_of_Figures.txt).

## Dependencies:

To reproduce the analysis and the results shown in this repository two conda / mamba environments are required. They can be found [here](https://github.com/EyringMLClimateGroup/behrens24james_SPCESM2_ML_ensembles/tree/main/environments):

1) The preprocessing environment (preprocessing_env.yml) uses these essential packages:
   - tensorflow=1.13.1 [https://github.com/tensorflow/tensorflow/releases/tag/v1.13.1](https://github.com/tensorflow/tensorflow/releases/tag/v1.13.1)
   - dask==2.1.0 [https://github.com/dask/dask/releases/tag/2.1.0](https://github.com/dask/dask/releases/tag/2.1.0)
   - keras==2.2.4 [https://github.com/keras-team/keras/releases/tag/2.2.4](https://github.com/keras-team/keras/releases/tag/2.2.4)
   - xarray==0.12.2 [https://github.com/pydata/xarray/releases/tag/v0.12.2]([https://github.com/pydata/xarray/releases/tag/v0.12.2)
  
  To enable a full functionality of the .cbrain code for preprocessing SPCESM2 data the use of the entire preprocessing environment is recommended. 

  ```
  mamba env create -f preprocessing_env.yml
  ```

2) The training and evaluation environment (training_evaluation_env.yml) uses these essential packages:
   - tensorflow=2.10.0 [https://github.com/tensorflow/tensorflow/releases/tag/v2.10.1](https://github.com/tensorflow/tensorflow/releases/tag/v2.10.1)
   - pytorch=1.12.1 [https://github.com/pytorch/pytorch/releases/tag/v1.12.1](https://github.com/pytorch/pytorch/releases/tag/v1.12.1)
   - keras=2.10.0 [https://github.com/keras-team/keras/releases/tag/v2.10.0](https://github.com/keras-team/keras/releases/tag/v2.10.0)
   - xarray=2022.3.0 [https://github.com/pydata/xarray/releases/tag/v2022.03.0](https://github.com/pydata/xarray/releases/tag/v2022.03.0)
   - dask=2022.5.2 [https://github.com/dask/dask/releases/tag/2022.05.2](https://github.com/dask/dask/releases/tag/2022.05.2)
   - xskillscore=0.0.24 [https://github.com/xarray-contrib/xskillscore/releases/tag/v0.0.24](https://github.com/xarray-contrib/xskillscore/releases/tag/v0.0.24)
  
  For a complete functionality of the code of this repository it is recommended to use the training and evaluation environment.

  ```
  mamba env create -f training_evaluation_env.yml
  ```

## Strategy for the reproduction of the results of the paper and the repository:

  ### Offline strategy:
  
  1) Build the mamba / conda environments detailed above
  2) Familiarize with the cbrain package with the quickstart guide that can be found here: https://github.com/raspstephan/CBRAIN-CAM/blob/master/quickstart.ipynb
  3) Download the data sets from zenodo :[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10598576.svg)](https://doi.org/10.5281/zenodo.10598576) or use raw SPCESM2 data
     
     3.1)  if SPCESM2 raw data is used: preprocess the data with the `preprocessing_real_geography.py` file, example configuration files can be found in the folder `pp_config` for training , validation, test and normalizataion data:

       ```
       python preprocessing_real_geography.py -c /pp_config/example_config_file.yml 
       ```
  4) Use the prepocessed example SPCESM2 data from zenodo or your own preprocessed SPCESM2 data for training all models
      
     4.1) Train all networks:
       The repective training files of individual ANNs can be found in the folders:   models/offline_models/ANNs_lin/ANN_*
       The repective training files of individual VEDs can be found in the folders:   models/offline_models/VEDs/VED_*
       The repective training files of ANN_dropout can be found in the folders:  models/offline_models/ANN_dropout

       For the training you need the preprocessed training, validation, test, normalization datasets and 1 single .nc file of SPCESM2 raw data to detrmine the vertical coordinate of the SPCESM2 model (variables hyai, hybi):

       ```
       python training file
       ```
     4.2) Transform all keras models with the conversion jupyter notebooks (*conversion*.py) into pytorch models (pytorch will be used for the rest of the offline evaluation)

       
  5) Run the deterministic_analysis Jupyter notebooks with all trained networks, here you need again the test data sets
     5.1) For the VED-varying ensemble please use VED_1 and the alpha_1.npy array that can be found in folder ('latent_perturbation_tuning')
     
  6) Run the uncertainty_quantification Jupyter notebooks (again use VED_1 and alpha_1 forr VED_varying)
  7) Run the CRPS_analysis notebooks

### Online strategy (HPC system needed):

  1) Download and compile CESM=2.1.3 from [GitHub](https://github.com/ESCOMP/CESM/releases/tag/release-cesm2.1.3) on HPC
  2) Fork the dedicated [Github repository](https://github.com/SciPritchardLab/CESM2-ML-coupler) and clone it on the HPC
  3) Adjust compilers of CESM2 to the used HPC
  4) Copy models/online_models folder containing FKB.txt files of ANNs and normalization files to HPC
     
     4.1) If you want to use your trained ANNs, please use models/online_models/fkb_keras_convert.py to convert .h5 files into .txt files for FKB ([Fortran-Keras-Bridge](https://github.com/scientific-computing/FKB))
        ```
        python fkb_keras_convert --weights_file ANN_*.h5 --output_file ANN_*.txt
        ```
     
  5) Copy Fortran run scipts in folder online_run_scripts to HPC
  6) Copy SPCESM2 initilization files to HPC
  7) Run CESM2 with the exmaple run scripts for indivudal ANNs and ensembles
     
     example:
     
     ```
     ./run_cesm2_frontera2.batch.partial-coupling.csh ANN_1 2013-02-01 
     ```
     use this in csh shell, the first command sets the runscript, the second one the used ANN in this case, the third one the initilization data of CESM2 run

     To enable an efficient parallezization on the HPC the use of e.g, parallel is recommented [https://www.gnu.org/software/parallel/](https://www.gnu.org/software/parallel/).




  8) Run the benchmark simulation of CESM2.1.3 with the Zhang-McLane convection scheme and the SP
     For SP the example run scripts can be used by commenting out the coupled variables of the ML scheme in the run scripts, then SP is fully used.

  9) Run the dedicated Jupyter notebooks of the online_evaluation folder based on the output of the simulations of CESM2 with the ML and tradional schemes

## License:

This code is released under MIT License. See [LICENSE](https://github.com/EyringMLClimateGroup/behrens24james_SPCESM2_ML_ensembles/blob/main/LICENSE) for more information.

  
     





