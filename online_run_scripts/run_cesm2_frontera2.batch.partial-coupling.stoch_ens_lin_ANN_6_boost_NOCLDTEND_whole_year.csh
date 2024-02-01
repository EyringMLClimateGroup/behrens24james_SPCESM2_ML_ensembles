#!/bin/csh
#
# this code reproduces the DNN-ens-CESM hybrid simulation for given model txt files and initial conditions of CESM2, author: Sungduk Yu, Gunnar Behrens
#run example:
# ./run_cesm2_frontera2.batch.partial-coupling.stochastic-ens.csh "'/models/online_models/ANN_1/ANN_1_CRM_lin_model.h5.txt','/models/online_models/ANN_2/ANN_2_CRM_lin_model.h5.txt','/models/online_models/ANN_3/ANN_3_CRM_lin_model.h5.txt','/models/online_models/ANN_4/ANN_4_CRM_lin_model.h5.txt','/models/online_models/ANN_5/ANN_5_CRM_lin_model.h5.txt','/models/online_models/ANN6/ANN_6_CRM_lin_model_boost.txt','/models/online_models/ANN_7/ANN_7_CRM_lin_model.h5.txt'" 2013-02-01

set job_script_path=`readlink -f $0`

set ANNs = $1
set RSTDATE = "$2"
set KVAR = $99

set run_time       = 08:00:00
set queue          = regular
set account        = UYLE0025
set pcount         = 1024
set pelayout       = /glade/work/gbehrens/CESM2-neuralnet-UCI/cheyenne/pelayout_cheyenne01.xml

## ====================================================================
#   define case
## ====================================================================

setenv REFDATE     $RSTDATE
setenv CCSMTAG     CESM2.1.3
setenv CASE        CESM2_ANNs_lin_ncpl-48_spcam-dt-20_${REFDATE}_partial-coupling-NOCLDTEND_ANN_6_boost_sENS_5_whole_year
setenv CASESET     HIST_CAM%SPCAMS_CLM50%SP_CICE%PRES_DOCN%DOM_RTM_SGLC_SWAV
setenv CASERES     f19_g17
setenv PROJECT     $account
setenv JOB_QUEUE   $queue
## ====================================================================
#   define directories <Please make sure the directories are correct>
## ====================================================================

setenv MACH      cheyenne
setenv CCSMROOT  /glade/work/gbehrens/$CCSMTAG
setenv CODEDIR   /glade/work/gbehrens/CESM2-neuralnet-UCI
setenv PTMP      $TMPDIR/cesm2_scratch
setenv CASEROOT  $PTMP/$CASE
setenv RUNDIR    $PTMP/$CASE/run
setenv EXEDIR    $PTMP/$CASE/build
setenv ARCHDIR   $PTMP/$CASE/archive
#setenv DATADIR   $SCRATCH/cesm2_scratch/inputdata # check
#setenv SCRATCH $PTMP

## ====================================================================
#  FKB file paths
## ====================================================================
set cb_inp_sub   = /models/online_models/norm_files/inp_sub.txt
set cb_inp_div   = /models/online_models/norm_files/inp_div.txt
set cb_out_scale = /models/online_models/norm_files/out_scale_CRM.txt

if ( ! -e ${cb_inp_sub} || ! -e ${cb_inp_div}  || ! -e ${cb_out_scale} ) then
  echo "One or more FKB txt files do not exist"
  exit 1
endif

## ====================================================================
#   create new case
## ====================================================================

#rm -rf $PTMP/$CASE
cd $CCSMROOT/cime/scripts
#cp $CODEDIR/pelayout* 
./create_newcase --case $CASEROOT  --pecount $pcount --pesfile $pelayout --compiler intel --mpilib mpt --res $CASERES --machine $MACH --compset $CASESET --run-unsupported #--input-dir $DATADIR 
cd  $CASEROOT

./xmlchange --append CAM_CONFIG_OPTS="-cppdefs '-DCBRAINDIAG -DBRAINDEBUG'" # CBRAINDIAG is necessary for 'partial coupling'
./xmlchange EXEROOT=$EXEDIR
./xmlchange RUNDIR=$RUNDIR
./xmlchange DOUT_S_ROOT=$ARCHDIR
./xmlchange --file env_workflow.xml --id JOB_QUEUE --val $queue
./xmlchange --file env_workflow.xml --id JOB_WALLCLOCK_TIME --val $run_time
./xmlchange --file env_run.xml --id RUN_STARTDATE --val "2003-03-01" # ignored in a branch run
./xmlchange --file env_run.xml --id STOP_N --val 12
./xmlchange --file env_run.xml --id STOP_OPTION --val "nmonths"
./xmlchange --file env_run.xml --id DOUT_S --val "FALSE"
./xmlchange --file env_run.xml --id RESUBMIT --val 0

# Restarting restart stuff (testing):
./xmlchange --file env_run.xml --id RUN_TYPE --val "branch" # won't allow change model time steps
# xmlchange --file env_run.xml --id RUN_TYPE --val "hybrid" # if timestep (ATM_NCPL) needs to be changed.
./xmlchange ATM_NCPL=48
./xmlchange --file env_run.xml --id RUN_REFDIR --val "/glade/work/sungduk/Gunnar/init_files/CESM2_ncpl-48_spcam-dt-20_sp-baseline/${REFDATE}-00000"
./xmlchange --file env_run.xml --id GET_REFCASE --val "TRUE"
./xmlchange --file env_run.xml --id RUN_REFCASE --val "CESM2_ncpl-48_spcam-dt-20_sp-baseline"
./xmlchange --file env_run.xml --id RUN_REFDATE --val "$REFDATE"
./xmlchange --file env_run.xml --id RUN_REFTOD --val "00000"

#  'QBCTEND', 'TBCTEND', 'CLDLIQBCTEND', 'CLDICEBCTEND', 'PRECSC', 'PRECC', 'FLWDS','NETSW','SOLL', 'SOLLD', 'SOLS', 'SOLSD'

cat <<EOF >! user_nl_cam
&spmd_fv_inparm
npr_yz = 32,4,4,32
/

&phys_ctl_nl
state_debug_checks = .true.
/

&cbrain_nl
inputlength  = 109
outputlength = 112
input_rh     = .false.
cb_inp_sub   = '${cb_inp_sub}'
cb_inp_div   = '${cb_inp_div}'
cb_out_scale = '${cb_out_scale}'
cb_partial_coupling = .true.
! cb_partial_coupling_vars =  '${KVAR}'
! cb_partial_coupling_vars =  'QBCTEND', 'TBCTEND', 'CLDLIQBCTEND', 'CLDICEBCTEND', 'PRECSC', 'PRECC', 'FLWDS','NETSW','SOLL', 'SOLLD', 'SOLS', 'SOLSD'
! cb_partial_coupling_vars =  'QBCTEND', 'TBCTEND', 'PRECSC', 'PRECC','NETSW','SOLL', 'SOLLD', 'SOLS', 'SOLSD'
cb_partial_coupling_vars =  'QBCTEND', 'TBCTEND', 'PRECSC', 'PRECC', 'FLWDS','NETSW','SOLL', 'SOLLD', 'SOLS', 'SOLSD'
! cb_partial_coupling_vars =  'QBCTEND', 'TBCTEND', 'CLDLIQBCTEND', 'CLDICEBCTEND' 
! cb_partial_coupling_vars =  'PRECSC', 'PRECC', 'FLWDS','NETSW','SOLL', 'SOLLD', 'SOLS', 'SOLSD'

cb_use_input_prectm1 = .true.

cb_do_ensemble = .true.
cb_ens_size = 7
cb_ens_fkb_model_list = ${ANNs}

cb_random_ens_size = 5

/

&cam_history_nl
fincl2 = 'T0:I', 'Q0:I', 'CLDLIQ0:I', 'CLDICE0:I', 'NETSW0:I', 'FLWDS0:I', 'PRECSC0:I', 'PRECSL0:I', 'PRECC0:I', 'PRECL0:I', 'SOLL0:I', 'SOLS0:I', 'SOLLD0:I', 'SOLSD0:I', 'PRECCdt0:I', 'PRECLdt0:I', 'LHFLX0:I', 'SHFLX0:I', 'SOLIN0:I', 'PS:I'
fincl3 = 'T1:I', 'Q1:I', 'CLDLIQ1:I', 'CLDICE1:I', 'NETSW1:I', 'FLWDS1:I', 'PRECSC1:I', 'PRECSL1:I', 'PRECC1:I', 'PRECL1:I', 'SOLL1:I', 'SOLS1:I', 'SOLLD1:I', 'SOLSD1:I'
fincl4 = 'T2:I', 'Q2:I', 'CLDLIQ2:I', 'CLDICE2:I', 'NETSW2:I', 'FLWDS2:I', 'PRECSC2:I', 'PRECSL2:I', 'PRECC2:I', 'PRECL2:I', 'SOLL2:I', 'SOLS2:I', 'SOLLD2:I', 'SOLSD2:I'
fincl5 = 'T3:I', 'Q3:I', 'CLDLIQ3:I', 'CLDICE3:I', 'NETSW3:I', 'FLWDS3:I', 'PRECSC3:I', 'PRECSL3:I', 'PRECC3:I', 'PRECL3:I', 'SOLL3:I', 'SOLS3:I', 'SOLLD3:I', 'SOLSD3:I'
nhtfrq = 0,1,1,1,1
mfilt  = 0,1,1,1,1
/
EOF

cd $CASEROOT

## ====================================================================
#   copy user source modifications
## ====================================================================
cp -v $CODEDIR/*.F90                   ./SourceMods/src.cam
cp -v $CODEDIR/namelist_definition.xml ./SourceMods/src.cam
cp -v ${job_script_path} ./job_submit_script.copy

## ====================================================================
#   setup / build / submit
## ====================================================================
./case.setup
./case.build
./case.submit

cd ..

