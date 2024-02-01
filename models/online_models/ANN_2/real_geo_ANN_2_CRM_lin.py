
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
import sherpa

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
intermediate_dim = 455

batch_size = 9402
batchsize=9402

in_vars = ['QBP', 'TBP','PS', 'SOLIN', 'SHFLX', 'LHFLX','PRECTt-dt','CLDLIQBP','CLDICEBP']
out_vars = ['QBCTEND','TBCTEND','CLDLIQBCTEND','CLDICEBCTEND','PREC_CRM_SNOW','PREC_CRM','NN2L_FLWDS','NN2L_DOWN_SW','NN2L_SOLL','NN2L_SOLLD','NN2L_SOLS','NN2L_SOLSD']

#loading the output normalization scalars for SP variables ( stds over 3 months of SP simulation)

scale_array=ps.read_csv('nn_config/scale_dicts/real_geography_SP_vars_updt.csv')


QBC_std_surf=scale_array.QBCTEND_std.values[-1]

TBC_std=scale_array.TBCTEND_std.values[-1]
CLDLIQBCTEND_std=scale_array.CLDLIQBCTEND_std.values[-1]
CLDICEBCTEND_std=scale_array.CLDICEBCTEND_std.values[-1]


PREC_CRM_SNOW_std=scale_array.PRECT_CRM_SNOW_std.values
PREC_CRM_std=scale_array.PRECT_CRM_std.values
FSNS_std=scale_array.FSNS_std.values
FSNT_std=scale_array.FSNT_std.values
FLNS_std=scale_array.FLNS_std.values
FLNT_std=scale_array.FLNT_std.values
NN2L_FLWDS_std=scale_array.NN2L_FLWDS_std.values
NN2L_DOWN_SW_std=scale_array.NN2L_DOWN_SW_std.values
NN2L_SOLL_std=scale_array.NN2L_SOLL_std.values
NN2L_SOLLD_std=scale_array.NN2L_SOLLD_std.values
NN2L_SOLS_std=scale_array.NN2L_SOLS_mean_std.values
NN2L_SOLSD_std=scale_array.NN2L_SOLSD_mean_std.values


# and the CAM variables 
scale_array_2D=ps.read_csv('nn_config/scale_dicts/real_geography_CESM_vars.csv')

TBP_std_surf=scale_array_2D.TBP_std.values[-1]

QBP_std_surf=scale_array_2D.QBP_std.values[-1]
CLDLIQBP_std=scale_array_2D.CLDLIQBP_std.values
CLDICEBP_std=scale_array_2D.CLDICEBP_std.values

Q_lat_std_surf=scale_array_2D.LHFLX_std.values

Q_sens_std_surf=scale_array_2D.SHFLX_std.values


Q_solar_std_surf=scale_array_2D.SOLIN_std.values

PS_std_surf=scale_array_2D.PS_std.values


# defining the scaling dict for the VAE training 

scale_dict_II = {
    'QBCTEND': 1/QBC_std_surf, 
    'QBP':1/QBP_std_surf,
    'TBCTEND': 1/TBC_std, 
    'CLDICEBCTEND': 1/CLDICEBCTEND_std, 
    'CLDLIQBCTEND': 1/CLDLIQBCTEND_std, 
    'CLDLIQBP':1/CLDLIQBP_std,
    'CLDICEBP':1/CLDICEBP_std,
    'TBP':1/TBP_std_surf,
    'FSNT': 1/FSNT_std, 
    'FSNS': 1/FSNS_std, 
    'FLNT': 1/FLNT_std, 
    'FLNS': 1/FLNS_std, 
    'NN2L_FLWDS':1/NN2L_FLWDS_std,
    'NN2L_DOWN_SW':1/NN2L_DOWN_SW_std,
    'NN2L_SOLL':1/NN2L_SOLL_std,
    'NN2L_SOLLD':1/NN2L_SOLLD_std,
    'NN2L_SOLS':1/NN2L_SOLS_std,
    'NN2L_SOLSD':1/NN2L_SOLSD_std,    
    'PREC_CRM': 1/PREC_CRM_std,
    'PREC_CRM_SNOW': 1/PREC_CRM_SNOW_std,
    'LHFLX': 1/Q_lat_std_surf, 
    'SHFLX': 1/Q_sens_std_surf, 
    'SOLIN': 1/Q_solar_std_surf,
    'PS':1/PS_std_surf
}







from cbrain.data_generator import DataGenerator

test_xr=xr.open_dataset('/p/home/jusers/behrens2/juwels/scratch_icon_a_ml/SPCESM_data/raw_data/CESM2_NN2_pelayout01_ens_07.cam.h1.2013-08-30-57600.nc')
hybi=test_xr.hybi
hyai=test_xr.hyai

PS = 1e5; P0 = 1e5;
P = P0*hyai+PS*hybi; # Total pressure [Pa]
dP = P[1:]-P[:-1];

strategy = tf.distribute.MirroredStrategy()
tf.config.list_physical_devices('GPU')



    
    

    
    
 # Takes representative value for PS since purpose is normalization


# defining training data (shuffled in space and time) consisting of July, August, September of  first year of SPCAM simulation

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


# defining validation data consisting of April, May and June of second year of SP simulations

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

    
with strategy.scope():
    
    
    input_lay=Input(shape=input_shape, name='encoder_input')
    x_0 =Dense(intermediate_dim, activation='elu')(input_lay)
    x_1 =Dense(intermediate_dim, activation='elu')(x_0)
    x_2 =Dense(intermediate_dim, activation='elu')(x_1)
    x_3 =Dense(intermediate_dim, activation='elu')(x_2)
    x_4 =Dense(intermediate_dim, activation='elu')(x_3)
    x_5 =Dense(intermediate_dim, activation='elu')(x_4)
    outputs_1= Dense(original_dim_output,activation='elu')(x_5)
    outputs_A = Dense(original_dim_output,activation='linear')(outputs_1)
    
    
    # instantiate encoder model
    ANN = Model(input_lay, outputs_A, name='ANN')
    ANN.summary()
    
    
    
    
    
    ANN.compile(ke.optimizers.Adam(lr=0.0003359780707654258), loss=mse, metrics=['mse','mae','accuracy'])

# set learning rate schedule
def schedule(epoch):
        

    if epoch < 7:
            
        return 0.0003359780707654258
    elif epoch < 14:
            
        return 0.0003359780707654258/5

    elif epoch < 21:
            
        return 0.0003359780707654258/25
        
    elif epoch < 28:
        return 0.0003359780707654258/125
        
    elif epoch < 35:
        return 0.0003359780707654258/625

    elif epoch<42:
        return 0.0003359780707654258/3125




        

callback_lr=LearningRateScheduler(schedule,verbose=1)
csv_logger = CSVLogger('real_geography/ANNs/ANN_2/ANN_2_CRM_lin.csv')
 
    
ANN.fit(train_gen,validation_data=(val_gen,None),epochs=40,shuffle=False,
        callbacks=[callback_lr,csv_logger])
    
    
    

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

np.save('real_geography/ANNs/ANN_2/ANN_2_CRM_lin',test_mse)
ANN.save_weights('real_geography/ANNs/ANN_2/ANN_2_CRM_lin.h5')
ANN.save('real_geography/ANNs/ANN_2/ANN_2_CRM_lin_model.h5')

ANN.save_weights('real_geography/ANNs/ANN_2/ANN_2_CRM_lin.tf')


