'''
Created on 27 de jan de 2019
@author: gusta
'''

from experiments.experiment_master import Experiment
from competitive_algorithms.pogmm_vrd import POGMM_VRD

# PARAMETROS USADOS
#def __init__(self, batch_size=200, num_models=5, P=10, pool_exclusion="pertinence", tax=0.2, pool_training=True, pool_reusing=True):
    
    
# 1. INSTANTIATE THE EXPERIMENT CLASS
EXP = Experiment(pasta='ICDM(GaussianVirtual)/Mechanisms')
    
    
###############################################################################################
########################### Parte 1 para corrigir o erro ######################################
###############################################################################################

# 2. INSTANTIATE THE MODELS TO BE RUNNED
#2.1 
Novo1 = POGMM_VRD(batch_size=200, pool_training=False, pool_reusing=False)
Novo1.NAME = "POGMM_VRD-Nothing"

# joining the models
MODELS = [Novo1]

#4. RUNNING THE MODELS CHOSEN
EXP.run(cross_validation=True, models=MODELS, datasets=[0, 2], executions=[0, 20])

###############################################################################################
###############################################################################################
###############################################################################################




###############################################################################################
################################# Parte 2 para continuar ###################################### 
###############################################################################################

# 2. INSTANTIATE THE MODELS TO BE RUNNED
#2.1 
Novo1 = POGMM_VRD(batch_size=200, pool_training=False, pool_reusing=False)
Novo1.NAME = "POGMM_VRD-Nothing"

#2.2 
Novo2 = POGMM_VRD(batch_size=200, pool_training=True, pool_reusing=False)
Novo2.NAME = "POGMM_VRD-Pool-training"

#2.3 
Novo3 = POGMM_VRD(batch_size=200, pool_training=True, pool_reusing=True)
Novo3.NAME = "POGMM_VRD-Reusing-Gaussians"

#2.2 
# joining the models
MODELS = [Novo1, Novo2, Novo3]

#4. RUNNING THE MODELS CHOSEN
EXP.run(cross_validation=True, models=MODELS, datasets=[2, 7], executions=[0, 20])

###############################################################################################
###############################################################################################
###############################################################################################


