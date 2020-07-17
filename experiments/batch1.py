'''
Created on 27 de jan de 2019
@author: gusta
'''

from experiments.experiment_master import Experiment
from competitive_algorithms.pogmm_vrd import POGMM_VRD

# PARAMETROS USADOS
#def __init__(self, batch_size=200, num_models=5, P=10, pool_exclusion="pertinence", tax=0.2, step=3, pool_training=True, pool_reusing=True):
    
# 1. INSTANTIATE THE EXPERIMENT CLASS
EXP = Experiment(pasta='ICDM(GaussianVirtual)/Batch')
    
# 2. INSTANTIATE THE MODELS TO BE RUNNED
#2.1
# batch 50 foi executado no experimento de series artificiais

#2.2 
Novo2 = POGMM_VRD(batch_size=100, step=2, pool_exclusion="oldest")
Novo2.NAME = "POGMM_VRD-batch=100"

#2.3 
Novo3 = POGMM_VRD(batch_size=150, step=2, pool_exclusion="oldest")
Novo3.NAME = "POGMM_VRD-batch=150"

#2.4
# batch 200 foi executado no experimento de series mecanismos

# joining the models
MODELS = [Novo2, Novo3]

#4. RUNNING THE MODELS CHOSEN
EXP.run(cross_validation=True, models=MODELS, datasets=[0, 7], executions=[0, 15])


