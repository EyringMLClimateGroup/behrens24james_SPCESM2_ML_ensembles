"""
This file contains functions used in the main preprocessing script.
Created on 2019-01-23-14-50
Author: Stephan Rasp, raspstephan@gmail.com
tgb - 11/13/2019 - Adding RH and deviation from moist adiabat
"""

from ..imports import *
from ..cam_constants import *
import pickle
import scipy.integrate as sin

# Set up logging, mainly to get timings easily.
logging.basicConfig(
    format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.DEBUG
)


# Define dictionary with vertical diffusion terms
diff_dict = {
    'TAP' : 'DTV',
    'QAP' : 'VD01'
}


def compute_bp(ds, var):
    """GCM state at beginning of time step before physics.
    ?BP = ?AP - physical tendency * dt
    Args:
        ds: entire xarray dataset
        var: BP variable name
    Returns:
        bp: xarray dataarray containing just BP variable, with the first time step cut.
    """
    base_var = var[:-2] + 'AP'
    return (ds[base_var] - ds[phy_dict[base_var]] * DT)[1:]  # Not the first time step

def compute_perc(ds, var, PERC_array, quantile_array):
    
    from scipy.interpolate import interp1d
    
    # Load variable from dataset
    var0 = ds[var[:-4]][1:]
    # Manually entering ranges for each variable: 
    # This should change and PERC_array should be a dictionary with variable names
    i0 = {}
    i0['PHQ'] = 94
    i0['TPHYSTND'] = 124
    i0['QRL'] = 154
    i0['QRS'] = 184
    
    # Project onto 1D percentile space to form the output
    output_percentile = np.zeros(var0.shape) # Initialization
    for ilev in range(var0.shape[1]):
        print('Interpolating level ',ilev,'out of ',var0.shape[1])
        interp_fx = interp1d(x=PERC_array[:,i0[var[:-4]]+ilev],y=quantile_array,bounds_error=False)
        output_percentile[:,ilev,:,:] = interp_fx(var0[:,ilev,:,:])
        
    return (output_percentile+0*var0**0) # Convert from numpy back to xarray

# tgb - 4/24/2021 - Real geograpny version taking into account new output vector shape
def compute_perc_RG(ds, var, PERC_array, quantile_array):
    
    from scipy.interpolate import interp1d
    
    # Load variable from dataset
    var0 = ds[var[:-4]][1:]
    # Manually entering ranges for each variable: 
    # This should change and PERC_array should be a dictionary with variable names
    i0 = {}
    i0['PTEQ'] = 64
    i0['PTTEND'] = 94
    i0['QRL'] = 124
    i0['QRS'] = 154
    
    # Project onto 1D percentile space to form the output
    output_percentile = np.zeros(var0.shape) # Initialization
    for ilev in range(var0.shape[1]):
        print('Interpolating level ',ilev,'out of ',var0.shape[1])
        interp_fx = interp1d(x=PERC_array[:,i0[var[:-4]]+ilev],y=quantile_array,bounds_error=False)
        output_percentile[:,ilev,:,:] = interp_fx(var0[:,ilev,:,:])
        
    return (output_percentile+0*var0**0) # Convert from numpy back to xarray

def compute_RH(ds):
    # tgb - 11/13/2019 - Calculates Relative humidity following notebook 027
    def RH(T,qv,P0,PS,hyam,hybm):
        R = 287
        Rv = 461
        p = (hyam*P0+hybm*PS).values # Total pressure (Pa)
        return Rv*p*qv/(R*esat(T))
    def esat(T):
        T0 = 273.16
        T00 = 253.16
        omega = np.maximum(0,np.minimum(1,(T-T00)/(T0-T00)))

        return (T>T0)*eliq(T)+(T<T00)*eice(T)+(T<=T0)*(T>=T00)*(omega*eliq(T)+(1-omega)*eice(T))
    def eliq(T):
        a_liq = np.array([-0.976195544e-15,-0.952447341e-13,0.640689451e-10,0.206739458e-7,0.302950461e-5,0.264847430e-3,0.142986287e-1,0.443987641,6.11239921]);
        c_liq = -80
        T0 = 273.16
        return 100*np.polyval(a_liq,np.maximum(c_liq,T-T0))
    def eice(T):
        a_ice = np.array([0.252751365e-14,0.146898966e-11,0.385852041e-9,0.602588177e-7,0.615021634e-5,0.420895665e-3,0.188439774e-1,0.503160820,6.11147274]);
        c_ice = np.array([273.15,185,-100,0.00763685,0.000151069,7.48215e-07])
        T0 = 273.16
        return (T>c_ice[0])*eliq(T)+\
    (T<=c_ice[0])*(T>c_ice[1])*100*np.polyval(a_ice,T-T0)+\
    (T<=c_ice[1])*100*(c_ice[3]+np.maximum(c_ice[2],T-T0)*(c_ice[4]+np.maximum(c_ice[2],T-T0)*c_ice[5]))
    
    TBP = compute_bp(ds,'TBP')
    QBP = compute_bp(ds,'QBP')
    
    return RH(TBP,QBP,ds['P0'],ds['PS'][1:,:,:],ds['hyam'],ds['hybm'])

# tgb - 2/25/2021 - Calculate saturation deficit
def compute_QSATdeficit(ds):
    
    def esat(T):
        T0 = 273.16
        T00 = 253.16
        omega = np.maximum(0,np.minimum(1,(T-T00)/(T0-T00)))

        return (T>T0)*eliq(T)+(T<T00)*eice(T)+(T<=T0)*(T>=T00)*(omega*eliq(T)+(1-omega)*eice(T))
    
    def eliq(T):
        a_liq = np.array([-0.976195544e-15,-0.952447341e-13,0.640689451e-10,0.206739458e-7,0.302950461e-5,0.264847430e-3,0.142986287e-1,0.443987641,6.11239921]);
        c_liq = -80
        T0 = 273.16
        return 100*np.polyval(a_liq,np.maximum(c_liq,T-T0))
    
    def eice(T):
        a_ice = np.array([0.252751365e-14,0.146898966e-11,0.385852041e-9,0.602588177e-7,0.615021634e-5,0.420895665e-3,0.188439774e-1,0.503160820,6.11147274]);
        c_ice = np.array([273.15,185,-100,0.00763685,0.000151069,7.48215e-07])
        T0 = 273.16
        return (T>c_ice[0])*eliq(T)+\
    (T<=c_ice[0])*(T>c_ice[1])*100*np.polyval(a_ice,T-T0)+\
    (T<=c_ice[1])*100*(c_ice[3]+np.maximum(c_ice[2],T-T0)*(c_ice[4]+np.maximum(c_ice[2],T-T0)*c_ice[5]))
    
    def qv(T,RH,P0,PS,hyam,hybm):
        R = 287
        Rv = 461
        p = (hyam*P0+hybm*PS).values # Total pressure (Pa)

        return R*esat(T)*RH/(Rv*p)

    def qsat(T,P0,PS,hyam,hybm):
        return qv(T,1,P0,PS,hyam,hybm)
    
    TBP = compute_bp(ds,'TBP')
    QBP = compute_bp(ds,'QBP')
    
    return qsat(TBP,ds['P0'],ds['PS'][1:,:,:],ds['hyam'],ds['hybm'])-QBP

# tgb - 2/25/2021 - Calculate buoyancy assuming adiabatic ascent
def compute_BCONS(ds):
    def esat(T):
        T0 = 273.16
        T00 = 253.16
        omega = np.maximum(0,np.minimum(1,(T-T00)/(T0-T00)))

        return (T>T0)*eliq(T)+(T<T00)*eice(T)+(T<=T0)*(T>=T00)*(omega*eliq(T)+(1-omega)*eice(T))
    def eliq(T):
        a_liq = np.array([-0.976195544e-15,-0.952447341e-13,0.640689451e-10,0.206739458e-7,0.302950461e-5,0.264847430e-3,0.142986287e-1,0.443987641,6.11239921]);
        c_liq = -80
        T0 = 273.16
        return 100*np.polyval(a_liq,np.maximum(c_liq,T-T0))
    def eice(T):
        a_ice = np.array([0.252751365e-14,0.146898966e-11,0.385852041e-9,0.602588177e-7,0.615021634e-5,0.420895665e-3,0.188439774e-1,0.503160820,6.11147274]);
        c_ice = np.array([273.15,185,-100,0.00763685,0.000151069,7.48215e-07])
        T0 = 273.16
        return (T>c_ice[0])*eliq(T)+\
    (T<=c_ice[0])*(T>c_ice[1])*100*np.polyval(a_ice,T-T0)+\
    (T<=c_ice[1])*100*(c_ice[3]+np.maximum(c_ice[2],T-T0)*(c_ice[4]+np.maximum(c_ice[2],T-T0)*c_ice[5]))
    
    def qv(T,RH,P0,PS,hyam,hybm):
        R = 287
        Rv = 461
        p = (hyam*P0+hybm*PS).values # Total pressure (Pa)

        return R*esat(T)*RH/(Rv*p)

    def qsat(T,P0,PS,hyam,hybm):
        return qv(T,1,P0,PS,hyam,hybm)
    
    def theta_e_calc(T,qv,P0,PS,hyam,hybm):
        p = (hyam*P0+hybm*PS).values # Total pressure (Pa)
        
        tmelt  = 273.15
        CPD = 1005.7
        CPV = 1870.0
        CPVMCL = 2320.0
        RV = 461.5
        RD = 287.04
        EPS = RD/RV
        ALV0 = 2.501E6
        
        r = qv / (1. - qv)
        # get ev in hPa 
        ev_hPa = 100*p*r/(EPS+r)
        #get TL
        TL = (2840. / ((3.5*np.log(T)) - (np.log(ev_hPa)) - 4.805)) + 55.
        #calc chi_e:
        chi_e = 0.2854 * (1. - (0.28*r))
        P0_norm = (P0/(hyam*P0+hybm*PS)).values

        theta_e = T * P0_norm**chi_e * np.exp(((3.376/TL) - 0.00254) * r * 1000. * (1. + (0.81 * r)))
        
        return theta_e

    def theta_e_sat_calc(T,qv,P0,PS,hyam,hybm):
        return theta_e_calc(T,qsat(T,P0,PS,hyam,hybm),P0,PS,hyam,hybm)
    
    TBP = compute_bp(ds,'TBP')
    QBP = compute_bp(ds,'QBP')
    
    return G*(theta_e_calc(TBP,QBP,ds['P0'],ds['PS'][1:,:,:],ds['hyam'],ds['hybm'])[:,-1,:,:]-\
              theta_e_sat_calc(TBP,QBP,ds['P0'],ds['PS'][1:,:,:],ds['hyam'],ds['hybm']))/\
theta_e_sat_calc(TBP,QBP,ds['P0'],ds['PS'][1:,:,:],ds['hyam'],ds['hybm'])   

# tgb - 06/15/2021 - Calculates MSE-based buoyancy following Fiaz's derivation
def compute_BMSE(ds):
    
    def esat(T):
        T0 = 273.16
        T00 = 253.16
        omega = np.maximum(0,np.minimum(1,(T-T00)/(T0-T00)))

        return (T>T0)*eliq(T)+(T<T00)*eice(T)+(T<=T0)*(T>=T00)*(omega*eliq(T)+(1-omega)*eice(T))
    def eliq(T):
        a_liq = np.array([-0.976195544e-15,-0.952447341e-13,0.640689451e-10,0.206739458e-7,0.302950461e-5,0.264847430e-3,0.142986287e-1,0.443987641,6.11239921]);
        c_liq = -80
        T0 = 273.16
        return 100*np.polyval(a_liq,np.maximum(c_liq,T-T0))
    def eice(T):
        a_ice = np.array([0.252751365e-14,0.146898966e-11,0.385852041e-9,0.602588177e-7,0.615021634e-5,0.420895665e-3,0.188439774e-1,0.503160820,6.11147274]);
        c_ice = np.array([273.15,185,-100,0.00763685,0.000151069,7.48215e-07])
        T0 = 273.16
        return (T>c_ice[0])*eliq(T)+\
    (T<=c_ice[0])*(T>c_ice[1])*100*np.polyval(a_ice,T-T0)+\
    (T<=c_ice[1])*100*(c_ice[3]+np.maximum(c_ice[2],T-T0)*(c_ice[4]+np.maximum(c_ice[2],T-T0)*c_ice[5]))
    
    def qv(T,RH,P0,PS,hyam,hybm):
        R = 287
        Rv = 461
        p = (hyam*P0+hybm*PS).values # Total pressure (Pa)

        return R*esat(T)*RH/(Rv*p)

    def qsat(T,P0,PS,hyam,hybm):
        return qv(T,1,P0,PS,hyam,hybm)
    
    TBP = compute_bp(ds,'TBP')
    QBP = compute_bp(ds,'QBP')
    
    QSAT0 = qsat(TBP,ds['P0'],ds['PS'][1:,:,:],ds['hyam'],ds['hybm'])
    
    def geopotential(T,qv,P0,PS,hyam,hybm):
        # Ideal gas law -> rho=p(R_d*T_v)
        eps = 0.622 # Ratio of molecular weight(H2O)/molecular weight(dry air)
        R_D = 287 # Specific gas constant of dry air in J/K/kg

        r = qv/(qv**0-qv)
        Tv = T*(r**0+r/eps)/(r**0+r)
        p = (hyam*P0+hybm*PS).values # Total pressure (Pa)
        RHO = p/(R_D*Tv)

        Z = -sin.cumtrapz(x=p,y=1/(G*RHO),axis=1)
        Z = np.concatenate((0*Z[:,0:1,:,:]**0,Z),axis=1)
        return Z-Z[:,[29],:,:]+2
    
    Rv = 461
    kappa = 1+(L_V**2)*QSAT0/(Rv*C_P*(TBP**2))
    
    z0 = geopotential(TBP,QBP,ds['P0'],ds['PS'][1:,:,:],ds['hyam'],ds['hybm'])
    
    h_plume = C_P*TBP[:,-1,:,:]+L_V*QBP[:,-1,:,:]
    h_satenv = C_P*TBP+L_V*QBP+G*z0 
    
    return (G/kappa)*(h_plume-h_satenv)/(C_P*TBP)

def compute_dRH_dt(ds):
# tgb - 12/01/2019 - Calculates dRH/dt following 027 and compute_bp
# tgb - 11/13/2019 - Calculates Relative humidity following notebook 027
    def RH(T,qv,P0,PS,hyam,hybm):
        R = 287
        Rv = 461
        p = (hyam*P0+hybm*PS).values # Total pressure (Pa)
        return Rv*p*qv/(R*esat(T))
    def esat(T):
        T0 = 273.16
        T00 = 253.16
        omega = np.maximum(0,np.minimum(1,(T-T00)/(T0-T00)))

        return (T>T0)*eliq(T)+(T<T00)*eice(T)+(T<=T0)*(T>=T00)*(omega*eliq(T)+(1-omega)*eice(T))
    def eliq(T):
        a_liq = np.array([-0.976195544e-15,-0.952447341e-13,0.640689451e-10,0.206739458e-7,0.302950461e-5,0.264847430e-3,0.142986287e-1,0.443987641,6.11239921]);
        c_liq = -80
        T0 = 273.16
        return 100*np.polyval(a_liq,np.maximum(c_liq,T-T0))
    def eice(T):
        a_ice = np.array([0.252751365e-14,0.146898966e-11,0.385852041e-9,0.602588177e-7,0.615021634e-5,0.420895665e-3,0.188439774e-1,0.503160820,6.11147274]);
        c_ice = np.array([273.15,185,-100,0.00763685,0.000151069,7.48215e-07])
        T0 = 273.16
        return (T>c_ice[0])*eliq(T)+\
    (T<=c_ice[0])*(T>c_ice[1])*100*np.polyval(a_ice,T-T0)+\
    (T<=c_ice[1])*100*(c_ice[3]+np.maximum(c_ice[2],T-T0)*(c_ice[4]+np.maximum(c_ice[2],T-T0)*c_ice[5]))
    
    TBP = compute_bp(ds,'TBP')
    QBP = compute_bp(ds,'QBP')
    
    return (RH(ds['TAP'][1:,:,:,:],ds['QAP'][1:,:,:,:],ds['P0'],ds['PS'][1:,:,:],ds['hyam'],ds['hybm'])-\
            RH(TBP,QBP,ds['P0'],ds['PS'][1:,:,:],ds['hyam'],ds['hybm']))/DT

def compute_TfromMA(ds):
    import pickle
    # tgb - 11/13/2019 - Calculate deviations from moist adiabatic profile following notebook 027
    pathPKL = '/local/Tom.Beucler/SPCAM_PHYS'
    hf = open(pathPKL+'20191113_MA.pkl','rb')
    MA = pickle.load(hf)
    
    T_MAfit = np.zeros((ds['lev'].size,ds['TS'][1:,:,:].values.flatten().size))
    for iTs,Ts in enumerate(ds['TS'][1:,:,:].values.flatten()):
        T_MAfit[:,iTs] = MA['Ts_MA'][:,np.abs(Ts-MA['Ts_range']).argmin()]
        
    T_MAfit_reshape = np.moveaxis(np.reshape(T_MAfit,(ds['lev'].size,
                                                      ds['TS'][1:,:,:].shape[0],
                                                      ds['TS'][1:,:,:].shape[1],
                                                      ds['TS'][1:,:,:].shape[2])),0,1)
    
    return compute_bp(ds,'TBP')-T_MAfit_reshape

def compute_TfromTS(ds):
    return compute_bp(ds,'TBP')-ds['TS'][1:,:,:]

def compute_TfromNS(ds):
    return compute_bp(ds,'TBP')-compute_bp(ds,'TBP')[:,-1,:,:]

def compute_LR(ds):
    
    C_P = 1.00464e3 # Specific heat capacity of air at constant pressure
    G = 9.80616 # Gravity constant
    
    def PI(PS,P0,hyai,hybi):    
        S = PS.shape
        return np.moveaxis(np.tile(P0,(31,S[1],S[2],1)),[0,1,2,3],[1,2,3,0]) *\
    np.moveaxis(np.tile(hyai,(S[1],S[2],1,1)),[0,1,2,3],[2,3,0,1]) + \
    np.moveaxis(np.tile(PS.values,(31,1,1,1)),0,1) * \
    np.moveaxis(np.tile(hybi,(S[1],S[2],1,1)),[0,1,2,3],[2,3,0,1])
    
    def rho(qv,T,PS,P0,hyam,hybm):
        eps = 0.622 # Ratio of molecular weight(H2O)/molecular weight(dry air)
        R_D = 287 # Specific gas constant of dry air in J/K/k

        r = qv/(qv**0-qv)
        Tv = T*(r**0+r/eps)/(r**0+r)

        S = Tv.shape
        p = np.moveaxis(np.tile(P0,(30,S[2],S[3],1)),[0,1,2,3],[1,2,3,0]) *\
        np.moveaxis(np.tile(hyam,(S[2],S[3],1,1)),[0,1,2,3],[2,3,0,1]) + \
        np.moveaxis(np.tile(PS.values,(30,1,1,1)),0,1) * \
        np.moveaxis(np.tile(hybm,(S[2],S[3],1,1)),[0,1,2,3],[2,3,0,1])

        return p/(R_D*Tv)
    
    PI_ds = PI(ds['PS'],ds['P0'],ds['hyai'],ds['hybi'])
    TI_ds = np.concatenate((compute_bp(ds,'TBP'),
                            np.expand_dims(ds['TS'][1:,:,:],axis=1)),axis=1)
    RHO_ds = rho(compute_bp(ds,'QBP'),compute_bp(ds,'TBP'),ds['PS'][1:,:,:],
                 ds['P0'][1:],ds['hyam'][1:,:],ds['hybm'][1:,:])
    
    return C_P*RHO_ds.values*(TI_ds[:,1:,:,:]-TI_ds[:,:-1,:,:])/\
(PI_ds[1:,1:,:,:]-PI_ds[1:,:-1,:,:])*\
ds['TAP'][1:,:,:,:]**0 # Multiplication by 1 to keep xarray attributes
# No need for it in custom tf layer

def compute_Carnotmax(ds):
    # tgb - 11/15/2019 - Calculates local Carnot efficiency from Tmin to Tmax = max(T) over z
    TBP = compute_bp(ds,'TBP')
    return -(TBP-TBP.max(axis=1))/(TBP.max(axis=1)-TBP.min(axis=1))

def compute_CarnotS(ds):
    # tgb - 11/15/2019 - Calculates local Carnot efficiency from Tmin to Tmax=Ts!=max(T) over z
    TBP = compute_bp(ds,'TBP')
    return -(TBP-ds['TS'][1:,:,:])/(ds['TS'][1:,:,:]-TBP.min(axis=1))

# tgb - 3/31/2020 - Adding (TNS-T)/(TNS-220K)
def compute_NSto220(ds):
    TBP = compute_bp(ds,'TBP')
    T_tropopause= 220 # Hardcode the tropopause temperature for now
    return (TBP[:,-1,:,:]-TBP)/(TBP[:,-1,:,:]-T_tropopause)

def compute_c(ds, base_var):
    """CRM state at beginning of time step before physics.
    ?_C = ?AP[t-1] - diffusion[t-1] * dt
    Note:
    compute_c() is the only function that returns data from the previous
    time step.
    Args:
        ds: xarray dataset
        base_var: Base variable to be computed
    Returns:
        c: xarray dataarray
    """
    c = ds[base_var].isel(time=slice(0, -1, 1))   # Not the last time step
    if base_var in diff_dict.keys():
        c -= ds[diff_dict[base_var]].isel(time=slice(0, -1, 1)) * DT
    # Change time coordinate. Necessary for later computation of adiabatic
    c['time'] = ds.isel(time=slice(1, None, 1))['time']
    return c

def compute_flux(ds,var):
    
    base_var = var[:-4]
    P = 1e5*(ds['hyai']+ds['hybi']); # Total pressure [Pa]
    dP = P[0,1:].values-P[0,:-1].values; # Differential pressure [Pa]
#     print('dP',dP.shape)
  
# tgb - 12/3/2019 - Commenting out these lines because SEF go so close to 0 that 
#     print('Base variable is ',base_var)
#     SEF = np.moveaxis(np.tile(ds['LHFLX'][1:,:,:]+ds['SHFLX'][1:,:,:],(ds[base_var].shape[1],1,1,1)),0,1)
#     dP = np.moveaxis(np.tile(dP,(SEF.shape[0],SEF.shape[2],SEF.shape[3],1)),3,1)
    
# #     print('ds[base_var]',ds[base_var].shape)
# #     print('SEF',SEF.shape)
# #     print('dP',dP.shape)
#     if base_var=='PHQ': return L_V*dP/G*ds[base_var][1:,:,:,:]/np.maximum(10,SEF)
#     elif base_var=='TPHYSTND': return C_P*dP/G*ds[base_var][1:,:,:,:]/np.maximum(10,SEF)

# tgb - 12/3/2019 - Divide by SEF fit based on latitude
    pathPKL = '/home/t/Tom.Beucler/SPCAM/CBRAIN-CAM/notebooks/tbeucler_devlog/PKL_DATA/'
    hf = open(pathPKL+'2019_12_03_SEF_fit.pkl','rb')
    SFfit = pickle.load(hf)
    lfit = SFfit['LHFlogfit']
    sfit = SFfit['SHFfit']
    
    x = np.log10((compute_bp(ds,'TBP')[:,-1,:,:]).values) # Temperature to define eps coordinate
    LHF = 10**(lfit[0]*x**0+lfit[1]*x**1+lfit[2]*x**2)
    LHF = np.moveaxis(np.tile(LHF,(ds[base_var].shape[1],1,1,1)),0,1)
#     print('LHFmean',np.mean(LHF))
#     print('LHFmax',np.max(LHF))
#     print('LHFmin',np.min(LHF))
    SHF = sfit*np.ones(LHF.shape)
    dP = np.moveaxis(np.tile(dP,(LHF.shape[0],LHF.shape[2],LHF.shape[3],1)),3,1)
#     print('PHQav',np.mean(L_V*dP/G*ds[base_var][1:,:,:,:].values))
    
    if base_var=='PHQ': return L_V*dP/G*ds[base_var][1:,:,:,:]/(LHF+SHF)
    elif base_var=='TPHYSTND': return C_P*dP/G*ds[base_var][1:,:,:,:]/(LHF+SHF)
    
def compute_eps(ds, var):
    # tgb - 11/26/2019 - Interpolates variable on epsilon grid

    # tgb - 11/26/2019 - Load data to project on eps grid
    pathPKL = '/home/t/Tom.Beucler/SPCAM/CBRAIN-CAM/notebooks/tbeucler_devlog/PKL_DATA/'
    hf = open(pathPKL+'2019_11_22_imin_TNS_logfit.pkl','rb')
    imfit = pickle.load(hf)['logmodel'][0]
    hf = open(pathPKL+'2019_11_22_eps_TNS_linfit.pkl','rb')
    epfit = pickle.load(hf)['linmodel']
    
    # Pre-process variables
    #eps_res = 100 # For now hardcode epsilon grid resolution
    eps_res = 30 # tgb - 11/29/2019 - For experiment 134
    # Extract variable name (assumes variable+EPS, e.g. 'TPHYSTNDEPS','PHQEPS','TBPEPS')
    base_var = var[:-3]
    print('Base variable is ',base_var)
    if 'BP' in base_var:
        da = compute_bp(ds, base_var)
    elif 'FLUX' in base_var:
        da = compute_flux(ds,base_var)
    elif base_var=='RH':
        da = compute_RH(ds)
    elif base_var=='dRHdt':
        da = compute_dRH_dt(ds)
    elif base_var=='TfromTS':
        da = compute_TfromTS(ds)
    else: da = ds[base_var][1:]
    daT = compute_bp(ds,'TBP') # Temperature to define eps coordinate

    # 1) Generate eps grid for the neural network with vertical resolution eps_res
    # and the interpolated input array
    eps_NN = np.linspace(0,1,eps_res)
    daI = np.reshape(np.moveaxis(da.values,1,3),(da.shape[0]*da.shape[2]*da.shape[3],30)) # Resized dataset
    daTI = np.reshape(np.moveaxis(daT.values,1,3),(daT.shape[0]*daT.shape[2]*daT.shape[3],30)) # Resized temperature
    x_interp = np.zeros((daI.shape[0],int(eps_res)))

    # 2) Calculates vertical interpolation domain [imin_eval:] and eps coordinate as a function of NS T = T[30]
    for isample in range(daI.shape[0]):
        #rint('isample=',isample,'/',daI.shape[0],'                                                          ',end='\r')
        x = daTI[isample,-1]
        #print('x=',x)
        imin_eval = int(np.rint(10**(imfit[0]*np.log10(x)**0+imfit[1]*np.log10(x)**1+\
                                     imfit[2]*np.log10(x)**2+imfit[3]*np.log10(x)**3+\
                                     imfit[4]*np.log10(x)**4)))
        #print('imin_eval=',imin_eval)
        eps_eval = epfit[:,0]*x**0+epfit[:,1]*x**1+epfit[:,2]*x**2+epfit[:,3]*x**3+epfit[:,4]*x**4
        # tgb - 11/23/2019 - Adds dummy 1 at the end because np.where evaluates y output even if condition false and not returning y
        eps_test = np.minimum(1,np.maximum(eps_eval,0))[imin_eval:]
        eps_eval = np.concatenate((np.minimum(1,np.maximum(eps_eval,0))[imin_eval:][::-1],[1]))

    # 3) Interpolate both T and q to the eps grid for the neural network
    # 3.1) Thermodynamic profiles to interpolate
    # tgb - 11/23/2019 - Adds dummy zero at the end because np.where evaluates y output even if condition false and not returning y
        x_input = np.concatenate((daI[isample,imin_eval:][::-1],[0]))
    # 3.2) Interpolation using searchsorted and low-level weighting implementation
    # The goal is to mimic T_interp = np.interp(x=eps_ref,xp=eps_eval,fp=T_input)
    # If left then T_input[0], if right then T_input[-1], else weighted average of T_input[iint-1] and T_input[iint]
        iint = np.searchsorted(eps_eval,eps_NN)
        x_interp[isample,:] = np.where(iint<1,x_input[0],np.where(iint>(30-imin_eval-1),x_input[30-imin_eval-1],\
                                                                  ((eps_eval[iint]-eps_NN)/\
                                                                   (eps_eval[iint]-eps_eval[iint-1]))*x_input[iint-1]+\
                                                                  ((eps_NN-eps_eval[iint-1])/\
                                                                   (eps_eval[iint]-eps_eval[iint-1]))*x_input[iint]))
    # 4) Converts data back to xarray of the right shape
    if eps_res==100:
        x_interp = np.moveaxis(np.reshape(x_interp,(da.shape[0],da.shape[2],da.shape[3],eps_res)),3,1)+\
        0*xr.concat((da,da,da,da[:,:10,:,:]),'lev')
    elif eps_res==30:
        x_interp = np.moveaxis(np.reshape(x_interp,(da.shape[0],da.shape[2],da.shape[3],eps_res)),3,1)+0*da
    
    x_interp.__setitem__('lev',eps_NN)
        
    return x_interp

# tgb - 3/1/2021 - Normalize LHF by near-surface specific humidity
def compute_LHF_nsQ(ds,eps):
    
    QBP = compute_bp(ds,'QBP')
    Qden = QBP[:,-1:,:,:]
    return ds['LHFLX'][:-1]/(L_V*np.maximum(eps,Qden))
    
# tgb - 3/1/2021 - Normalize LHF by near-surface specific humidity contrast
def compute_LHF_nsDELQ(ds,eps):
    
    def esat(T):
        T0 = 273.16
        T00 = 253.16
        omega = np.maximum(0,np.minimum(1,(T-T00)/(T0-T00)))

        return (T>T0)*eliq(T)+(T<T00)*eice(T)+(T<=T0)*(T>=T00)*(omega*eliq(T)+(1-omega)*eice(T))
    
    def eliq(T):
        a_liq = np.array([-0.976195544e-15,-0.952447341e-13,0.640689451e-10,0.206739458e-7,0.302950461e-5,0.264847430e-3,0.142986287e-1,0.443987641,6.11239921]);
        c_liq = -80
        T0 = 273.16
        return 100*np.polyval(a_liq,np.maximum(c_liq,T-T0))
    
    def eice(T):
        a_ice = np.array([0.252751365e-14,0.146898966e-11,0.385852041e-9,0.602588177e-7,0.615021634e-5,0.420895665e-3,0.188439774e-1,0.503160820,6.11147274]);
        c_ice = np.array([273.15,185,-100,0.00763685,0.000151069,7.48215e-07])
        T0 = 273.16
        return (T>c_ice[0])*eliq(T)+\
    (T<=c_ice[0])*(T>c_ice[1])*100*np.polyval(a_ice,T-T0)+\
    (T<=c_ice[1])*100*(c_ice[3]+np.maximum(c_ice[2],T-T0)*(c_ice[4]+np.maximum(c_ice[2],T-T0)*c_ice[5]))
    
    def qv(T,RH,P0,PS,hyam,hybm):
        R = 287
        Rv = 461
        p = (hyam*P0+hybm*PS).values # Total pressure (Pa)

        return R*esat(T)*RH/(Rv*p)

    def qsat(T,P0,PS,hyam,hybm):
        return qv(T,1,P0,PS,hyam,hybm)
    
    QBP = compute_bp(ds,'QBP')
    Qden = qsat(ds['TS'][1:,:,:],ds['P0'],ds['PS'][1:,:,:],ds['hyam'][:,-1],ds['hybm'][:,-1])-QBP[:,-1,:,:].values
    return ds['LHFLX'][:-1]/(L_V*np.maximum(eps,Qden))
    
# tgb - 3/1/2021 - Normalize SHF by near-surface temperature contrast
def compute_SHF_nsDELT(ds,eps):
    print('Computing TBP')
    TBP = compute_bp(ds,'TBP')
    print('Computing Tden')
    Tden = ds['TS'][1:,:,:]-TBP
    print('Returning normalized SHF')
    return ds['SHFLX'][:-1]/(C_P*np.maximum(eps,Tden))

def compute_adiabatic(ds, base_var):
    """Compute adiabatic tendencies.
    Args:
        ds: xarray dataset
        base_var: Base variable to be computed
    Returns:
        adiabatic: xarray dataarray
    """
    return (compute_bp(ds, base_var) - compute_c(ds, base_var)) / DT

def load_O3_AQUA(ds):
    """Load O3 profile for aquaplanet simulation
    """
    dT = compute_bp(ds,'TBP') # Calculate TBP to get the right shape
    
    path_O3 = '/export/nfs0home/tbeucler/CBRAIN-CAM/notebooks/tbeucler_devlog/PKL_DATA/2021_01_24_O3.pkl' # Hardcode O3 path for now
    dO3 = pickle.load(open(path_O3,'rb'))
    O3 = dO3['O3_aqua']
    O3_interpolated = O3.interp({"lev": dT.lev}) # Interpolate to the right vertical levels
    # Reshape to TBP's shape
    O3_reshaped = np.moveaxis(np.tile(O3_interpolated,(dT.shape[0],dT.shape[-1],1,1)),source=[1,2],destination=[3,2]) 
    
    return (O3_reshaped+0*dT**0) # Make sure it is in xarray Dataset format with right dimensions

def create_stacked_da(ds, vars, PERC_array = None, quantile_array = None, real_geography = None):
    """
    In this function the derived variables are computed and the right time steps are selected.
    Parameters
    ----------
    ds: mf_dataset with dimensions [time, lev, lat, lon]
    vars: list of input and output variables
    Returns
    -------
    da: dataarray with variables [vars, var_names]
    """
    var_list, names_list = [], []
    for var in vars:
        print('var is ',var)
        #if 'EPS' in var:
        #    da = compute_eps(ds, var)
        #elif 'FLUX' in var:
        #    da = compute_flux(ds,var)
        #elif 'BP' in var:
        #    if real_geography:
        #        da = ds[var][1:]
        #    else:
        #        da = compute_bp(ds, var)
        #elif 'PERC' in var:
        #    print('PERC script: var is ',var,'real_geography is ',real_geography)
        #    if real_geography and var=='PHQPERC':
        #        da = compute_perc_RG(ds,'PTEQPERC', PERC_array, quantile_array)
        #    elif real_geography and var=='TPHYSTNDPERC':
        #        da = compute_perc_RG(ds,'PTTENDPERC', PERC_array, quantile_array)
        #    elif real_geography:
        #        da = compute_perc_RG(ds, var, PERC_array, quantile_array)
        #    else:
        #        da = compute_perc(ds, var, PERC_array, quantile_array)
        if var in ['LHFLX', 'SHFLX']:
            da = ds[var][:-1]
        elif var == 'PRECST':
            da = (ds['PRECSC'] + ds['PRECSL'])[1:]
        elif real_geography and var == 'PRECT':
            da = (ds['NN2L_PRECC'] + ds['NN2L_PRECL'])[1:]
        elif real_geography and var == 'PREC_SNOW':
            da = (ds['NN2L_PRECSC']+ ds['NN2L_PRECSL'])[1:]
        elif real_geography and var == 'PREC_CRM':
            da = ds['NN2L_PRECC'][1:]
        elif real_geography and var == 'PREC_CRM_SNOW':
            da = ds['NN2L_PRECSC'][1:]    
        elif var == 'PRECT':
            da = (ds['PRECC'] + ds['PRECL'])[1:]
        #elif var == 'RH':
        #    da = compute_RH(ds)
        #elif var == 'QSATdeficit':
        #    da = compute_QSATdeficit(ds)
        #elif var == 'dRHdt':
        #    da = compute_dRH_dt(ds)
        #elif var == 'TfromMA':
        #    da = compute_TfromMA(ds)
        #elif var == 'Carnotmax':
        #    da = compute_Carnotmax(ds)
        #elif var == 'CarnotS':
        #    da = compute_CarnotS(ds)
        #elif var == 'TfromTS':
        #    da = compute_TfromTS(ds)
        #elif var == 'TfromNS':
        #    da = compute_TfromNS(ds)
        #elif var == 'T_NSto220':
        #    da = compute_NSto220(ds)
        #elif var == 'BCONS':
        #    da = compute_BCONS(ds)
        #elif var == 'BMSE':
        #    da = compute_BMSE(ds)
        #elif var == 'LR':
        #    da = compute_LR(ds)
        #elif var == 'EPTNS':
        #    da = compute_EPTNS(ds)
        #elif var == 'LHF_nsQ':
        #    da = compute_LHF_nsQ(ds,1e-3) # For now, hardcode eps=1e-3
        #elif var == 'LHF_nsDELQ':
        #    da = compute_LHF_nsDELQ(ds,1e-3) # For now, hardcode eps=1e-3
        #elif var == 'SHF_nsDELT':
        #    da = compute_SHF_nsDELT(ds,1) # For now, hardcore eps=1K
        #elif var == 'O3_AQUA':
        #    da = load_O3_AQUA(ds)
        #elif var == 'TPHYSTND500':
        #    if real_geography: da = ds['PTTEND'][1:,[18],:,:]
        #    else: da = ds['TPHYSTND'][1:,[18],:,:]
        elif real_geography and var == 'PHQ':
            da = ds['PTEQ'][1:]
        ##GB    
        elif real_geography and var == 'PHQ_ICE':
            da = ds['PTECLDICE'][1:]
        elif real_geography and var == 'PHQ_LIQ':
            da = ds['PTECLDLIQ'][1:]
            
        ##GB 
        elif real_geography and var == 'TPHYSTND':
            da = ds['PTTEND'][1:]
        elif real_geography and var == 'QBCTEND':
            da = (ds['QBC'] - ds['QBP'])[1:]/1800
        elif real_geography and var == 'CLDLIQBCTEND':
            da = (ds['CLDLIQBC'] - ds['CLDLIQBP'])[1:]/1800
        elif real_geography and var == 'CLDICEBCTEND':
            da = (ds['CLDICEBC'] - ds['CLDICEBP'])[1:]/1800    
            
            
        elif real_geography and var == 'TBCTEND':
            da = (ds['TBC'] - ds['TBP'])[1:]/1800
            
            
        elif real_geography and var == 'CRM_ICE':
            da = ds['SPQI'][1:]
        elif real_geography and var == 'CRM_LIQ':
            da = ds['SPQC'][1:]
        elif real_geography and var == 'NN2L_DOWN_SW':
            da = ds['NN2L_NETSW'][1:]    
        ##    
        elif 'dt_adiabatic' in var:
            base_var = var[:-12] + 'AP'
            da = compute_adiabatic(ds, base_var)
        elif 't-dt' in var:
            if real_geography and var == 'PRECTt-dt':
                da = (ds['NN2L_PRECC'] + ds['NN2L_PRECL'])[:-1]
            if real_geography and var == 'PRECT_SNOWt-dt':
                da = (ds['NN2L_PRECSC'] + ds['NN2L_PRECSL'])[:-1]    
                
            if real_geography and var == 'TPHYSTNDt-dt':
                da = ds['PTTEND'][:-1]
            if real_geography and var == 'PHQt-dt':
                da = ds['PTEQ'][:-1]    
            if real_geography and var == 'PHQ_ICEt-dt':
                da = ds['PTECLDICE'][:-1]
            if real_geography and var == 'PHQ_LIQt-dt':
                da = ds['PTECLDLIQ'][:-1]
            if real_geography and var == 'CRM_ICEt-dt':
                da = ds['SPQI'][:-1]    
            if real_geography and var == 'CRM_LIQt-dt':
                da = ds['SPQC'][:-1]     
            #elif var == 'PRECTt-dt':
            #    da = (ds['PRECC'] + ds['PRECL'])[:-1]
            #else: da = ds[var[:-4]][:-1]
        else:
            da = ds[var][1:]
        var_list.append(da)
        nlev = da.lev.size if 'lev' in da.coords else 1
        names_list.extend([var] * nlev)

    concat_da = rename_time_lev_and_cut_times(ds, var_list)

    # Delete unused coordinates and set var_names as coordinates
    concat_da['var_names'] = np.array(names_list).astype('object')
    #names_da = xr.DataArray(names_list, coords=[concat_da.coords['stacked']])
    a = 3
    return concat_da


def rename_time_lev_and_cut_times(ds, da_list):
    """Create new time and lev coordinates and cut times for non-cont steps
    This is a bit of a legacy function. Should probably be revised.
    Args:
        ds: Merged dataset
        da_list: list of dataarrays
    Returns:
        da, name_da: concat da and name da
    """

    ilev = 0
    for da in da_list:
        da.coords['time'] = np.arange(da.coords['time'].size)
        if 'lev' in da.coords:
            da.coords['lev'] = np.arange(ilev, ilev + da.coords['lev'].size)
            ilev += da.coords['lev'].size
        else:
            da.expand_dims('lev')
            da.coords['lev'] = ilev
            ilev += 1

    # Concatenate
    da = xr.concat(da_list, dim='lev')
    # Cut out time steps
    cut_time_steps = np.where(np.abs(np.diff(ds.time)) > 2.09e-2)[0]
    clean_time_steps = np.array(da.coords['time'])
    print('These time steps are cut:', cut_time_steps)
    clean_time_steps = np.delete(clean_time_steps, cut_time_steps)
    da = da.isel(time=clean_time_steps)
    # Rename
    da = da.rename({'lev': 'var_names'})
    da = da.rename('vars')

    return da


def reshape_da(da):
    """
    Parameters
    ----------
    da: dataarray with [time, stacked, lat, lon]
    Returns
    -------
    da: dataarray with [sample, stacked]
    """
    da = da.stack(sample=('time', 'lat', 'lon'))
    return da.transpose('sample', 'var_names')


def preprocess(in_dir, in_fns, out_dir, out_fn, vars, lev_range=(0, 30), path_PERC = None, real_geography = False):
    """
    This is the main script that preprocesses one file.
    Returns
    -------
    """
    
    if in_dir=='None': logging.debug(f'No in_dir so in_fns is set to in_fns')
    else: in_fns = path.join(in_dir, in_fns)
    out_fn = path.join(out_dir, out_fn)
    logging.debug(f'Start preprocessing file {out_fn}')

    logging.info('Reading input files')
    logging.debug(f'Reading input file {in_fns}')
    ds = xr.open_mfdataset(in_fns, decode_times=False, decode_cf=False, concat_dim='time')
    
    # tgb - 3/28/2021 - Added this line to read percentiles from PKL file
    if path_PERC:
        logging.info('Reading PKL file containing univariate distribution of output variables')
        logging.info(f'Reading {path_PERC}')
        hf = open(path_PERC,'rb')
        tmp = pickle.load(hf)
        PERC_array = tmp['PERC_array']
        quantile_array = tmp['quantile_array']                   
    
    logging.info('Crop levels')
    ds = ds.isel(lev=slice(*lev_range, 1))

    logging.info('Create stacked dataarray')
    logging.info(f'Real geography flag set to {real_geography}')    
    if real_geography:
        logging.info('Earth-like simulation detected')
        if path_PERC:
            da = create_stacked_da(ds, vars, PERC_array = PERC_array, quantile_array = quantile_array, real_geography=real_geography)
        else:
            da = create_stacked_da(ds, vars, real_geography=real_geography)
    else:
        logging.info('Aquaplanet simulation detected')
        if path_PERC:
            da = create_stacked_da(ds, vars, PERC_array = PERC_array, quantile_array = quantile_array)
        else:
            da = create_stacked_da(ds, vars)

    logging.info('Stack and reshape dataarray')
    da = reshape_da(da).reset_index('sample')

    logging.info(f'Save dataarray as {out_fn}')
    da.load().to_netcdf(out_fn)

    logging.info('Done!')


if __name__ == '__main__':
    fire.Fire(preprocess)
    
def preprocess_list(in_dir, out_dir, out_fn, vars, list_xr1, list_xr2, lev_range=(0, 30)):
    """
    This is the main script that preprocesses one file.
    Returns
    -------
    """
    
    out_fn = path.join(out_dir, out_fn)
    logging.debug(f'Start preprocessing file {out_fn}')

    logging.info('Reading input files')
    logging.debug(f'Reading input file {list_xr1}')
    logging.debug(f'and {list_xr2}')
    ds = xr.open_mfdataset([list_xr1,list_xr2], decode_times=False, decode_cf=False, concat_dim='time')

    logging.info('Crop levels')
    ds = ds.isel(lev=slice(*lev_range, 1))

    logging.info('Create stacked dataarray')
    da = create_stacked_da(ds, vars)

    logging.info('Stack and reshape dataarray')
    da = reshape_da(da).reset_index('sample')

    logging.info(f'Save dataarray as {out_fn}')
    da.to_netcdf(out_fn)

    logging.info('Done!')