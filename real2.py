'''
Created on 27 de jan de 2019
@author: gusta
'''

from experiments.experiment_master import Experiment
from competitive_algorithms.pogmm_vrd import POGMM_VRD

# PARAMETROS USADOS
#def __init__(self, batch_size=200, num_models=5, P=10, pool_exclusion="pertinence", tax=0.2, step=3, pool_training=True, pool_reusing=True):

# 1. INSTANTIATE THE EXPERIMENT CLASS
EXP = Experiment(pasta='ICDM2/Real')
    
# 2. INSTANTIATE THE MODELS TO BE RUNNED
MODELS = [POGMM_VRD(batch_size=200, metric="gmean")]

#4. RUNNING THE MODELS CHOSEN
EXP.run(cross_validation=True, models=MODELS, datasets=[9, 10], executions=[10, 20])




