'''
Created on 27 de jan de 2019
@author: gusta
'''

from experiments.experiment_master import Experiment
from competitive_algorithms.pogmm_vrd import POGMM_VRD

# PARAMETROS USADOS
#def __init__(self, batch_size=100, num_models=5, P=10, pool_exclusion="pertinence", tax=0.2, step=3, pool_training=True, pool_reusing=True):


# 1. INSTANTIATE THE EXPERIMENT CLASS
EXP = Experiment(pasta='ICDM/Pool Exclusion')
    
# 2. INSTANTIATE THE MODELS TO BE RUNNED
#2.1 
Novo1 = POGMM_VRD(pool_exclusion="older")
Novo1.NAME = "POGMM_VRD-older"

#2.2 
Novo2 = POGMM_VRD(pool_exclusion="distance")
Novo2.NAME = "POGMM_VRD-distance"

#2.4 
Novo3 = POGMM_VRD(pool_exclusion="pertinence")
Novo3.NAME = "POGMM_VRD-pertinence"

# joining the models
MODELS = [Novo1, Novo2, Novo3]

#4. RUNNING THE MODELS CHOSEN
EXP.run(cross_validation=True, models=MODELS, datasets=[0, 7], executions=[0, 15])


