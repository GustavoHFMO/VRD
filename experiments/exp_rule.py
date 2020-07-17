'''
Created on 27 de jan de 2019
@author: gusta
'''

from experiments.experiment_master import Experiment
from competitive_algorithms.agmm import AGMM

# 1. INSTANTIATE THE EXPERIMENT CLASS
EXP = Experiment(pasta='rule')
    
# 2. INSTANTIATE THE MODELS TO BE RUNNED
#2.1 
Novo1 = AGMM(virtual=True, recorrencia=False)
Novo1.NAME = "min"
Novo1.CLASSIFIER.rule = "min"

#2.1 
Novo2 = AGMM(virtual=True, recorrencia=False)
Novo2.NAME = "max"
Novo2.CLASSIFIER.rule = "max"

#2.1 
Novo3 = AGMM(virtual=True, recorrencia=False)
Novo3.NAME = "mean"
Novo3.CLASSIFIER.rule = "mean"

#3. DEFINING THE MODELS THAT ARE GOING TO BE EXECUTED
MODELS = [Novo1, Novo2, Novo3]

#4. RUNNING THE MODELS CHOSEN
EXP.run(cross_validation=True, models=MODELS, datasets=[0, 6], executions=[0, 30])


