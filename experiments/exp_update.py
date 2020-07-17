'''
Created on 27 de jan de 2019
@author: gusta
'''

from experiments.experiment_master import Experiment
from competitive_algorithms.agmm import AGMM

# 1. INSTANTIATE THE EXPERIMENT CLASS
EXP = Experiment(pasta='update')
    
# 2. INSTANTIATE THE MODELS TO BE RUNNED
#2.1 
Novo1 = AGMM(virtual=True, recorrencia=False)
Novo1.NAME = "none"
Novo1.CLASSIFIER.tipo_atualizacao = "none"

#2.1 
Novo2 = AGMM(virtual=True, recorrencia=False)
Novo2.NAME = "all"
Novo2.CLASSIFIER.tipo_atualizacao = "all"

#2.1 
Novo3 = AGMM(virtual=True, recorrencia=False)
Novo3.NAME = "period"
Novo3.CLASSIFIER.tipo_atualizacao = "period"

#2.1
Novo4 = AGMM(virtual=True, recorrencia=False)
Novo4.NAME = "correct"
Novo4.CLASSIFIER.tipo_atualizacao = "correct"

#2.1
Novo5 = AGMM(virtual=True, recorrencia=False)
Novo5.NAME = "error"
Novo5.CLASSIFIER.tipo_atualizacao = "error"

#3. DEFINING THE MODELS THAT ARE GOING TO BE EXECUTED
MODELS = [Novo1, Novo2, Novo3, Novo4, Novo5]

#4. RUNNING THE MODELS CHOSEN
EXP.run(cross_validation=True, models=MODELS, datasets=[0, 6], executions=[0, 30])


