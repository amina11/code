cmake_minimum_required(VERSION 2.8)
project(KVAR NONE)
#####neet to modify it

### Setup for run on local machine or on CUI clusters
option(USE_CUI_CLUSTER "On CUI Cluster?" ON)
#set(NODE_LIST  bluewhale01 CACHE STRING "Which nodes to run on")
#CUI Cluster use SLURM and hardcoded matlab path.
if(${USE_CUI_CLUSTER})
	set(MATLAB /opt/uau/matlab2014a/bin/matlab)
	set(SLURM srun -t 2880 --partition=batch --exclude=bluewhale03,omura)
else()
#Use matlab simply and no SLURM.
	set(MATLAB matlab)
	set(SLURM bash)
endif()


### Setup for local TS directories
set(TS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/..  CACHE PATH "Main TS directory")
set(TS_BIN_DIR ${TS_DIR}/..  CACHE PATH "function files directory")
set(OUT_FOLDER  CV_lambda_total_500h_100h_500i  CACHE STRING "Output data folder")
set(TS_DIR_CURRENT  ${CMAKE_CURRENT_SOURCE_DIR} CACHE PATH "current directory")
set(TS_DATA_DIR ${TS_DIR_CURRENT}/${OUT_FOLDER})
set(FUNCTION_NAME  matlabTesting.sh CACHE STRING "function name")
### Control variables for individual methods
#set(Met_Name ADMM CACHE STRING "Name of method to test")
#set(Outer_CVs 1 CACHE STRING "Number of outer cv samples")
set(Inner_CVs 3  CACHE STRING "Number of inner cv samples")
set(Reg_Type1 1 2 4 6 8 10   CACHE VAR "Number of regularization types - 0 if makes no sense for method")
set(Reg_Type2 0 0.001 0.01  0.1 1 10  CACHE VAR "Number of regularization types - 0 if makes no sense for method")

#########????
### replace ; with nothing just for the touch file
#string(REGEX REPLACE ";" "" RTypes "${Reg_Types}")
#string(REGEX REPLACE ";" "" KTypes "${Kernel_Types}")

### ????? Configure the jobs matlab*.sh from matlab*.sh.in for this
###???configure_file(${TS_BIN_DIR}/matlabTesting.sh.in ${TS_BIN_DIR}/matlabTesting.sh @ONLY)


### Main loop for fitting a model accross all inner cv samples
set(tTesting)
foreach(reg_num1 ${Reg_Type1})
foreach(reg_num2 ${Reg_Type2})
foreach(inn_num RANGE 1 ${Inner_CVs})

set(data_dir ${TS_DATA_DIR}/total${reg_num1}/lambda2${reg_num2}/Inner${inn_num})
add_custom_command(OUTPUT ${data_dir}/error.mat COMMAND ${SLURM} ${TS_BIN_DIR}/${FUNCTION_NAME}  ${TS_DIR_CURRENT}/datapartition/${inn_num}/data.mat  ${data_dir} ${reg_num1} ${reg_num2})

list(APPEND tTesting ${data_dir}/error.mat)
endforeach(inn_num RANGE 1 ${Inner_CVs})
endforeach(reg_num2 ${Reg_Type2})
endforeach(reg_num1 ${Reg_Type1})


###??? Create targets and generate control files ###create dependency
##add_custom_command(OUTPUT doneTesting COMMAND touch ${TS_DATA_DIR}/blabla DEPENDS ${tTesting})
add_custom_target(doTesting ALL DEPENDS ${tTesting})   ##create goal and tell the file do job which is called done testing
