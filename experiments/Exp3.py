'''
Created on 27 de jan de 2019
@author: gusta
'''

from experiments.experiment_master import Experiment
from competitive_algorithms.novo import NOVO
from detectors.no_detector import noDetector

# 1. INSTANTIATE THE EXPERIMENT CLASS
EXP = Experiment(pasta='analiseMecanismos')
    
# 2. INSTANTIATE THE MODELS TO BE RUNNED
#2.1 
Novo1 = NOVO(virtual=False, real=False)
Novo1.NAME = "batch"
Novo1.DETECTOR = noDetector()

#2.1 
Novo2 = NOVO(virtual=False, real=True)
Novo2.NAME = "semVirtual"

#2.1 
Novo3 = NOVO(virtual=True, real=False)
Novo3.NAME = "semReal"
Novo3.DETECTOR = noDetector()

#3. DEFINING THE MODELS THAT ARE GOING TO BE EXECUTED
MODELS = [Novo1, Novo2, Novo3]

#4. RUNNING THE MODELS CHOSEN
EXP.run(cross_validation=False, models=MODELS, datasets=[0, 7], executions=[0, 30])


