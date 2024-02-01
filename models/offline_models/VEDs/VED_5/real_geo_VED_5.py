"""
train and test data of VED 5
author: G.Behrens

"""
from tensorflow.keras.layers import Input, Dense, Concatenate, Lambda 
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



# reparameterization trick of VAE 
# instead of sampling from Q(z|X), sample epsilon = N(0,I)
# z = z_mean + sqrt(var) * epsilon
def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.

    # Arguments
        args (tensor): mean and log of variance of Q(z|X)

    # Returns
        z (tensor): sampled latent vector
    based on VAE presented on keras webpage for keras version 1 /
    recent keras VAE version can be seen on
    https://keras.io/examples/generative/vae/
    """

    z_mean, z_log_var = args
    batch= K.shape(z_mean)[0]
    dim=K.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon=K.random_normal(shape=(batch,dim)) # epsilion= random_normal distributed tensor
    sample_prob=z_mean+K.exp(0.5*z_log_var)*epsilon #exp= elementwise exponential
    return sample_prob


original_dim_input=109  # CBRAIN input node size

original_dim_output=int(113) # CBRAIN output node size 


# network parameters
latent_dim=13
intermediate_dim = 337    
batch_size = 4624     
kl_weight=6.805253892490205e-05
lr_init=0.0010127432508693162
act='elu'

# define some shapes 

input_shape = (original_dim_input,)
decoder_input_shape=(latent_dim,)
out_shape=(original_dim_output,)

# define input variables 
in_vars = ['QBP', 'TBP','PS', 'SOLIN', 'SHFLX', 'LHFLX','PRECTt-dt','CLDLIQBP','CLDICEBP']
# define output variables 
out_vars = ['QBCTEND','TBCTEND','CLDLIQBCTEND','CLDICEBCTEND', 'PRECT','PREC_CRM_SNOW','PREC_CRM',
            'NN2L_FLWDS','NN2L_DOWN_SW','NN2L_SOLL','NN2L_SOLLD','NN2L_SOLS','NN2L_SOLSD']

#loading the output normalization scalars for SP variables ( stds over 2 months of SP simulation)

scale_array=ps.read_csv('nn_config/scale_dicts/real_geography_SP_vars_updt.csv')


QBC_std_surf=scale_array.QBCTEND_std.values[-1]

TBC_std=scale_array.TBCTEND_std.values[-1]
CLDLIQBCTEND_std=scale_array.CLDLIQBCTEND_std.values[-1]
CLDICEBCTEND_std=scale_array.CLDICEBCTEND_std.values[-1]


PRECT_std=scale_array.PRECT_std.values
PREC_CRM_SNOW_std=scale_array.PRECT_CRM_SNOW_std.values
PREC_CRM_std=scale_array.PRECT_CRM_std.values

NN2L_FLWDS_std=scale_array.NN2L_FLWDS_std.values
NN2L_DOWN_SW_std=scale_array.NN2L_DOWN_SW_std.values
NN2L_SOLL_std=scale_array.NN2L_SOLL_std.values
NN2L_SOLLD_std=scale_array.NN2L_SOLLD_std.values
NN2L_SOLS_std=scale_array.NN2L_SOLS_mean_std.values
NN2L_SOLSD_std=scale_array.NN2L_SOLSD_mean_std.values



# defining the scaling dict for the VED training 

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
    'PRECT': 1/PRECT_std,
    'PREC_CRM': 1/PREC_CRM_std,
    'PREC_CRM_SNOW': 1/PREC_CRM_SNOW_std
}







from cbrain.data_generator import DataGenerator
# transform vertical coords in pressure levels 
test_xr=xr.open_dataset('/p/home/jusers/behrens2/juwels/scratch_icon_a_ml/SPCESM_data/raw_data/CESM2_NN2_pelayout01_ens_07.cam.h1.2013-08-30-57600.nc')
hybi=test_xr.hybi
hyai=test_xr.hyai

PS = 1e5; P0 = 1e5;
P = P0*hyai+PS*hybi; # Total pressure [Pa]
dP = P[1:]-P[:-1];
# enable multi-GPU training 
strategy = tf.distribute.MirroredStrategy()
tf.config.list_physical_devices('GPU')



# defining training data (shuffled in space and time) consisting of 7 consecutive days from all months of the first year of SPCESM simulation

train_gen = DataGenerator(
        data_fn = '/p/home/jusers/behrens2/juwels/scratch_icon_a_ml/SPCESM_data/2013_train_7_consec_days_shuffle.nc',
        input_vars = in_vars,
        output_vars = out_vars,
        norm_fn = '/p/home/jusers/behrens2/juwels/scratch_icon_a_ml/SPCESM_data/2013_norm_7_consec_days.nc',
        input_transform = ('mean', 'maxrs'),
        output_transform = scale_dict_II,
        batch_size=batch_size,
        shuffle=True
        )


# defining validation data consisting of 7 consecutive days from all months of the second year of SPCESM simulations

val_gen = DataGenerator(
        data_fn = '/p/home/jusers/behrens2/juwels/scratch_icon_a_ml/SPCESM_data/2014_val_7_consec_days.nc',
        input_vars = in_vars,
        output_vars = out_vars,
        norm_fn = '/p/home/jusers/behrens2/juwels/scratch_icon_a_ml/SPCESM_data/2013_norm_7_consec_days.nc',
        input_transform = ('mean', 'maxrs'),
        output_transform = scale_dict_II,
        batch_size=batch_size,
        shuffle=True
        )

    
with strategy.scope():
    
    
    input_lay=Input(shape=input_shape, name='encoder_input')
    x_0 =Dense(intermediate_dim, activation=act)(input_lay)
    x_1 =Dense(intermediate_dim, activation=act)(x_0)
    x_2 =Dense(intermediate_dim/2, activation=act)(x_1)
    x_3 =Dense(intermediate_dim/4, activation=act)(x_2)
    x_4 =Dense(intermediate_dim/8, activation=act)(x_3)
    x_5 =Dense(intermediate_dim/16, activation=act)(x_4)
    z_mean = Dense(latent_dim, name='z_mean')(x_5)
    z_log_var = Dense(latent_dim, name='z_log_var')(x_5)



    # reparametrization trick
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

    # instantiate encoder model
    encoder = Model(input_lay, [z_mean, z_log_var, z], name='encoder')
    encoder.summary()
    
    input_decoder=Input(shape=decoder_input_shape, name='decoder_input')
    x_0 =Dense(intermediate_dim/16, activation=act)(input_decoder)
    x_1 =Dense(intermediate_dim/8, activation=act)(x_0)
    x_2 =Dense(intermediate_dim/4, activation=act)(x_1)
    x_3 =Dense(intermediate_dim/2, activation=act)(x_2)
    x_4 =Dense(intermediate_dim, activation=act)(x_3)
    x_5 =Dense(intermediate_dim, activation=act)(x_4)


    outputs_1= Dense(original_dim_output,activation=act)(x_5)
    outputs_A = Dense(104,activation='linear')(outputs_1)
    outputs_B = Dense(9,activation='relu')(outputs_1)
    outputs= Concatenate()([outputs_A,outputs_B])
 
    
    # instantiate decoder model
    decoder = Model(input_decoder, outputs, name='decoder')
    decoder.summary()
    decoder_outputs=decoder(encoder(input_lay)[2])
    VED=Model(input_lay,decoder_outputs, name='VED')
    VED.summary()
    
    #define VED-loss
    
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    VAE_loss = K.mean(kl_loss*kl_weight)
    VED.add_loss(VAE_loss)
    VED.add_metric(kl_loss, name='kl_loss', aggregation='mean')
    
    
    #compile VED
    
    VED.compile(ke.optimizers.Adam(lr=lr_init), loss=mse, metrics=['mse','mae','accuracy'])

# set learning rate schedule
def schedule(epoch):
        

    if epoch < 7:
            
        return lr_init
    elif epoch < 14:
            
        return lr_init/5

    elif epoch < 21:
            
        return lr_init/25
        
    elif epoch < 28:
        return lr_init/125
        
    elif epoch < 35:
        return lr_init/625

    elif epoch<42:
        return lr_init/3125


# save history 
        

callback_lr=LearningRateScheduler(schedule,verbose=1)
csv_logger = CSVLogger('real_geography/VEDs/VED_5/VED_5.csv')
 
# fit VED     
VED.fit(train_gen,validation_data=(val_gen,None),epochs=40,shuffle=False,
        callbacks=[callback_lr,csv_logger])
    
    
#load test data and compute test_mse     

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

pred=np.zeros((2000,int(96*144),113))
true=np.zeros((2000,int(96*144),113))
for i in tqdm(np.arange(2000)):
    pred[i]=VED.predict(val_gen_II[i][0])
    true[i]=val_gen_II[i][1]

pred_=np.reshape(pred,(int(2000*96*144),113))
true_=np.reshape(true,(int(2000*96*144),113))

del pred, true

test_mse=np.mean((pred_[:]-true_[:])**2,0)

#save VED 5 files 
np.save('real_geography/VEDs/VED_5/VED_5',test_mse)
VED.save_weights('real_geography/VEDs/VED_5/VED_5_weights.h5')
VED.save_weights('real_geography/VEDs/VED_5/VED_5.tf')
VED.save_model('real_geography/VEDs/VED_5/VED_5_model.h5')
