"""
train and test script for ANN 6
author: G.Behrens
"""


from tensorflow.keras.layers import Input, Dense
from cbrain.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.losses import mse
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import LearningRateScheduler, CSVLogger


import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import xarray as xr

import tensorflow.keras as ke


from cbrain.imports import *

from cbrain.utils import *

from cbrain.cam_constants import *

from cbrain.data_generator import DataGenerator
import pandas as ps



original_dim_input=109  # CBRAIN input node size

original_dim_output=int(112) # CBRAIN output node size 


# network parameters
input_shape = (original_dim_input,)
out_shape=(original_dim_output,)
intermediate_dim = 433

batch_size = 11162

batchsize=11162

#define input vars
in_vars = ['QBP', 'TBP','PS', 'SOLIN', 'SHFLX', 'LHFLX','PRECTt-dt','CLDLIQBP','CLDICEBP']
#and output vars 
out_vars = ['QBCTEND','TBCTEND','CLDLIQBCTEND','CLDICEBCTEND','PREC_CRM_SNOW','PREC_CRM','NN2L_FLWDS','NN2L_DOWN_SW','NN2L_SOLL','NN2L_SOLLD','NN2L_SOLS','NN2L_SOLSD']

#loading the output normalization scalars for SP variables ( stds over 2 months of SP simulation)

scale_array=ps.read_csv('nn_config/scale_dicts/real_geography_SP_vars_updt.csv')


QBC_std_surf=scale_array.QBCTEND_std.values[-1]

TBC_std=scale_array.TBCTEND_std.values[-1]
CLDLIQBCTEND_std=scale_array.CLDLIQBCTEND_std.values[-1]
CLDICEBCTEND_std=scale_array.CLDICEBCTEND_std.values[-1]


PREC_CRM_SNOW_std=scale_array.PRECT_CRM_SNOW_std.values
PREC_CRM_std=scale_array.PRECT_CRM_std.values

NN2L_FLWDS_std=scale_array.NN2L_FLWDS_std.values
NN2L_DOWN_SW_std=scale_array.NN2L_DOWN_SW_std.values
NN2L_SOLL_std=scale_array.NN2L_SOLL_std.values
NN2L_SOLLD_std=scale_array.NN2L_SOLLD_std.values
NN2L_SOLS_std=scale_array.NN2L_SOLS_mean_std.values
NN2L_SOLSD_std=scale_array.NN2L_SOLSD_mean_std.values





# defining the scaling dict for the ANN training 

scale_dict_II = {
    'QBCTEND': 1/QBC_std_surf, 
    'TBCTEND': 1/TBC_std, 
    'CLDICEBCTEND': 1/CLDICEBCTEND_std, 
    'CLDLIQBCTEND': 1/CLDLIQBCTEND_std, 
    'NN2L_FLWDS':1/NN2L_FLWDS_std,
    'NN2L_DOWN_SW':1/NN2L_DOWN_SW_std,
    'NN2L_SOLL':1/NN2L_SOLL_std,
    'NN2L_SOLLD':1/NN2L_SOLLD_std,
    'NN2L_SOLS':1/NN2L_SOLS_std,
    'NN2L_SOLSD':1/NN2L_SOLSD_std,    
    'PREC_CRM': 1/PREC_CRM_std,
    'PREC_CRM_SNOW': 1/PREC_CRM_SNOW_std
}







from cbrain.data_generator import DataGenerator

# get vertical axis in pressure coords
test_xr=xr.open_dataset('/p/home/jusers/behrens2/juwels/scratch_icon_a_ml/SPCESM_data/raw_data/CESM2_NN2_pelayout01_ens_07.cam.h1.2013-08-30-57600.nc')
hybi=test_xr.hybi
hyai=test_xr.hyai

PS = 1e5; P0 = 1e5;
P = P0*hyai+PS*hybi; # Total pressure [Pa]
dP = P[1:]-P[:-1];

# enable multi-GPU training
strategy = tf.distribute.MirroredStrategy()
tf.config.list_physical_devices('GPU')



    
    

    
    

# defining training data (shuffled in space and time) consisting of first 7 consecutive days of each month of year 2013 of SPCAM simulation

train_gen = DataGenerator(
        data_fn = '/p/home/jusers/behrens2/juwels/scratch_icon_a_ml/SPCESM_data/2013_train_7_consec_days_shuffle.nc',
        input_vars = in_vars,
        output_vars = out_vars,
        norm_fn = '/p/home/jusers/behrens2/juwels/scratch_icon_a_ml/SPCESM_data/2013_norm_7_consec_days.nc',
        input_transform = ('mean', 'maxrs'),
        output_transform = scale_dict_II,
        batch_size=batchsize,
        shuffle=True
        )


# defining validation data consisting of first 7 consecutive days of each month of year 2014 of SP simulations

val_gen = DataGenerator(
        data_fn = '/p/home/jusers/behrens2/juwels/scratch_icon_a_ml/SPCESM_data/2014_val_7_consec_days.nc',
        input_vars = in_vars,
        output_vars = out_vars,
        norm_fn = '/p/home/jusers/behrens2/juwels/scratch_icon_a_ml/SPCESM_data/2013_norm_7_consec_days.nc',
        input_transform = ('mean', 'maxrs'),
        output_transform = scale_dict_II,
        batch_size=batchsize,
        shuffle=True
        )

# build ANN
with strategy.scope():
    
    
    input_lay=Input(shape=input_shape, name='encoder_input')
    x_0 =Dense(intermediate_dim, activation='elu')(input_lay)
    x_1 =Dense(intermediate_dim, activation='elu')(x_0)
    x_2 =Dense(intermediate_dim, activation='elu')(x_1)
    x_3 =Dense(intermediate_dim, activation='elu')(x_2)
    x_4 =Dense(intermediate_dim, activation='elu')(x_3)
    

    outputs_1= Dense(original_dim_output,activation='elu')(x_4)
    outputs_A = Dense(original_dim_output,activation='linear')(outputs_1)
   
    # instantiate ANN model
    ANN = Model(input_lay, outputs_A, name='ANN')
    ANN.summary()
    
    
    
    
    
    ANN.compile(ke.optimizers.Adam(lr=0.0004723136556966778), loss=mse, metrics=['mse','mae','accuracy'])

# set learning rate schedule
def schedule(epoch):
        

    if epoch < 7:
            
        return 0.0013733334054815
    elif epoch < 14:
            
        return 0.0013733334054815/5

    elif epoch < 21:
            
        return 0.0013733334054815/25
        
    elif epoch < 28:
        return 0.0013733334054815/125
        
    elif epoch < 35:
        return 0.0013733334054815/625

    elif epoch<42:
        return 0.0013733334054815/3125


#save history        

callback_lr=LearningRateScheduler(schedule,verbose=1)
csv_logger = CSVLogger('real_geography/ANNs/ANN_6/ANN_6_CRM_lin.csv')
 
     
ANN.fit(train_gen,validation_data=(val_gen,None),epochs=40,shuffle=False,
        callbacks=[callback_lr,csv_logger])
    
    
    
# load test data and compute test mse
val_gen_II = DataGenerator(
    data_fn = '/p/home/jusers/behrens2/juwels/scratch_icon_a_ml/SPCESM_data/2015_test_7_consec_days_mem.nc',
    input_vars = in_vars,
    output_vars = out_vars,
    norm_fn = '/p/home/jusers/behrens2/juwels/scratch_icon_a_ml/SPCESM_data/2013_norm_7_consec_days.nc',
    input_transform = ('mean', 'maxrs'),
    output_transform = scale_dict_II,
    batch_size=int(96*144),
    shuffle=True
)

pred=np.zeros((2000,int(96*144),112))
true=np.zeros((2000,int(96*144),112))
for i in tqdm(np.arange(2000)):
    pred[i]=ANN.predict(val_gen_II[i][0])
    true[i]=val_gen_II[i][1]

pred_=np.reshape(pred,(int(2000*96*144),112))
true_=np.reshape(true,(int(2000*96*144),112))

del pred, true

test_mse=np.mean((pred_[:]-true_[:])**2,0)

#save ANN files

np.save('real_geography/ANNs/ANN_6/ANN_6_CRM_lin',test_mse)
ANN.save_weights('real_geography/ANNs/ANN_6/ANN_6_CRM_lin.h5')
ANN.save('real_geography/ANNs/ANN_6/ANN_6_CRM_lin_model.h5')

ANN.save_weights('real_geography/ANNs/ANN_6/ANN_6_CRM_lin.tf')


