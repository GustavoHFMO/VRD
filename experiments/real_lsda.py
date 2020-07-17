'''
Created on 27 de jan de 2019
@author: gusta
'''

from experiments.experiment_master import Experiment
from competitive_algorithms.ldd_dsda import LDD_DSDA

# PARAMETROS USADOS
#def __init__(self, batch_size=200, num_models=5, P=10, pool_exclusion="pertinence", tax=0.2, step=3, pool_training=True, pool_reusing=True):

# 1. INSTANTIATE THE EXPERIMENT CLASS
EXP = Experiment(pasta='ICDM(GaussianVirtual)/Real')
    
# 2. INSTANTIATE THE MODELS TO BE RUNNED
MODELS = [LDD_DSDA(train_size=200)]

#4. RUNNING THE MODELS CHOSEN
EXP.run(cross_validation=True, models=MODELS, datasets=[7, 10], executions=[0, 20])
