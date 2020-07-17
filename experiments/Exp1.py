'''
Created on 27 de jan de 2019
@author: gusta
'''
from experiments.experiment_master import Experiment
from competitive_algorithms.gmm_vrd import GMM_VRD
from competitive_algorithms.igmmcd import IGMM_CD
from competitive_algorithms.dynse import Dynse
from competitive_algorithms.agmm import AGMM

# 1. INSTANTIATE THE EXPERIMENT CLASS
EXP = Experiment(pasta='AAA')
    
# 2. INSTANTIATE THE MODELS TO BE RUNNED
#2.1.
GMMVRD = GMM_VRD()
    
#2.2.
DYNSE = Dynse()
    
#2.3
IGMMCD = IGMM_CD()

#2.4
Novo = AGMM()

#3. DEFINING THE MODELS THAT ARE GOING TO BE EXECUTED
MODELS = [GMMVRD, DYNSE, IGMMCD, Novo]

#4. RUNNING THE MODELS CHOSEN
EXP.run(cross_validation=True, models=MODELS, datasets=[0, 10], executions=[0, 1])


