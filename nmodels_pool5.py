'''
Created on 27 de jan de 2019
@author: gusta
'''

from experiments.experiment_master import Experiment
from competitive_algorithms.pogmm_vrd import POGMM_VRD

# PARAMETROS USADOS
#def __init__(self, batch_size=200, num_models=5, P=10, pool_exclusion="older", tax=0.2, pool_training=True, pool_reusing=True):


# 1. INSTANTIATE THE EXPERIMENT CLASS
EXP = Experiment(pasta='ICDM(GaussianVirtual)/NP')

P = [10]
for i in P:    
    # 2. INSTANTIATE THE MODELS TO BE RUNNED
    #2.1 
    Novo1 = POGMM_VRD(batch_size=200, num_models=1, P=i)
    Novo1.NAME = "POGMM_VRD2-NUM=1"+"-P="+str(i)
    
    #2.2 
    Novo2 = POGMM_VRD(batch_size=200, num_models=3, P=i)
    Novo2.NAME = "POGMM_VRD2-NUM=3"+"-P="+str(i)
    
    #2.3
    Novo3 = POGMM_VRD(batch_size=200, num_models=5, P=i)
    Novo3.NAME = "POGMM_VRD2-NUM=5"+"-P="+str(i)
    
    #2.5
    Novo4 = POGMM_VRD(batch_size=200, num_models=7, P=i)
    Novo4.NAME = "POGMM_VRD2-NUM=7"+"-P="+str(i)
    
    # joining the models
    MODELS = [Novo1, Novo2, Novo3, Novo4]
    
    #4. RUNNING THE MODELS CHOSEN
    EXP.run(cross_validation=True, models=MODELS, datasets=[6, 7], executions=[0, 20])


