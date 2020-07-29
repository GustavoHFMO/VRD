'''
Created on 27 de jan de 2019
@author: gusta
'''

from experiments.experiment_master import Experiment
from competitive_algorithms.pogmm_vrd import POGMM_VRD
from competitive_algorithms.ogmmf_vrd import OGMMF_VRD

# PARAMETROS USADOS
#def __init__(self, batch_size=200, num_models=5, P=10, pool_exclusion="pertinence", tax=0.2, step=3, pool_training=True, pool_reusing=True):

# 1. INSTANTIATE THE EXPERIMENT CLASS
EXP = Experiment(pasta='ICDM (Gmean)/Artificial')
    
# 2. INSTANTIATE THE MODELS TO BE RUNNED
MODELS = [OGMMF_VRD(batch_size=50, metric="gmean"), POGMM_VRD(batch_size=50, metric="gmean")]

#4. RUNNING THE MODELS CHOSEN
EXP.run(cross_validation=True, models=MODELS, datasets=[0, 7], executions=[0, 10])




