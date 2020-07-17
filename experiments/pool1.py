'''
Created on 27 de jan de 2019
@author: gusta
'''

from experiments.experiment_master import Experiment
from competitive_algorithms.pogmm_vrd import POGMM_VRD

# PARAMETROS USADOS
#def __init__(self, batch_size=100, num_models=5, P=10, pool_exclusion="pertinence", tax=0.2, step=3, pool_training=True, pool_reusing=True)

# 1. INSTANTIATE THE EXPERIMENT CLASS
EXP = Experiment(pasta='ICDM(GaussianVirtual)/Pool Length')
    
# 2. INSTANTIATE THE MODELS TO BE RUNNED
#2.1 
Novo1 = POGMM_VRD(batch_size=200, P=5, step=2, pool_exclusion="oldest")
Novo1.NAME = "POGMM_VRD-P=5"

#2.2 
# foi executado no experimento mechanisms

#2.3
Novo3 = POGMM_VRD(batch_size=200, P=15, step=2, pool_exclusion="oldest")
Novo3.NAME = "POGMM_VRD-P=15"

#2.4
Novo4 = POGMM_VRD(batch_size=200, P=20, step=2, pool_exclusion="oldest")
Novo4.NAME = "POGMM_VRD-P=20"

# joining the models
MODELS = [Novo1, Novo3, Novo4]

#4. RUNNING THE MODELS CHOSEN
EXP.run(cross_validation=True, models=MODELS, datasets=[0, 7], executions=[0, 15])


