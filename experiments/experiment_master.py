'''
Created on 5 de out de 2018
@author: gusta
'''

from table_generator.excel_table import Tabela_excel
from streams.readers.arff_reader import ARFFReader
from competitive_algorithms.gmm_vrd import GMM_VRD
from competitive_algorithms.igmmcd import IGMM_CD
from competitive_algorithms.dynse import Dynse
from filters.project_creator import Project
import pandas as pd
import copy
import time

class Experiment():
    def __init__(self, pasta):
        self.pasta = pasta
    
    def chooseDataset(self, number):
        if(number==0):
            labels, _, stream_records = ARFFReader.read("data_streams/_synthetic/circles.arff")
            name = 'circles'
        
        elif(number==1):
            labels, _, stream_records = ARFFReader.read("data_streams/_synthetic/sine1.arff")
            name = 'sine1'
        
        elif(number==2):
            labels, _, stream_records = ARFFReader.read("data_streams/_synthetic/sine2.arff")
            name = 'sine2'
        
        elif(number==3):
            labels, _, stream_records = ARFFReader.read("data_streams/_synthetic/virtual_5changes.arff")
            name = 'virtual_5changes'
        
        elif(number==4):
            labels, _, stream_records = ARFFReader.read("data_streams/_synthetic/virtual_9changes.arff")
            name = 'virtual_9changes'
        
        elif(number==5):
            labels, _, stream_records = ARFFReader.read("data_streams/_synthetic/SEA.arff")
            name = 'SEA'
        
        elif(number==6):
            labels, _, stream_records = ARFFReader.read("data_streams/_synthetic/SEARec.arff")
            name = 'SEARec'
            
        elif(number==7):
            labels, _, stream_records = ARFFReader.read("data_streams/real/elec.arff")
            name = 'elec'
        
        elif(number==8):
            labels, _, stream_records = ARFFReader.read("data_streams/real/noaa.arff")
            name = 'noaa'
        
        elif(number==9):
            labels, _, stream_records = ARFFReader.read("data_streams/real/PAKDD.arff")
            name = 'PAKDD'
        
        print(name)
        return name, labels, stream_records
        
    def saveInformation(self, i, xxx, patch, tb_accuracy, tb_time, dataset_name, model_name, predictions, target, accuracy, time):
    
        # storing the prediction
        df = pd.DataFrame(data={'predictions': predictions, 'target': target})
        df.to_csv(patch+model_name+"-"+dataset_name+"_"+str(i)+".csv")
            
        # storing the accuracy of system 
        tb_accuracy.Adicionar_dado(0, i+1, xxx, accuracy)
        
        # storing the time of system 
        tb_time.Adicionar_dado(0, i+1, xxx, time)
        
        # printing
        print(model_name, ': ', accuracy)
        
    def run(self, cross_validation, models, datasets, executions):
        
        # patch to store the archive experiments
        pjt = Project("projects", self.pasta)
        patch = pjt.get_path() 
        
        # defining the names that will fill the sheet
        cabecalho = [i.NAME for i in models]
        
        # for to iterate above the datasets
        for i in range(datasets[0], datasets[1]):
            
            # for to iterate above the executions
            for j in range(executions[0], executions[1]):
                
                # choosing the dataset
                dataset_name, labels, stream_records = self.chooseDataset(i)
                
                # table to store only the accuracy of models        
                if(j == executions[0]):
                    tb_accuracy = Tabela_excel()
                    tb_accuracy.Criar_tabela(nome_tabela=patch+dataset_name+'-accuracy', 
                                             folhas=['modelos'], 
                                             cabecalho=cabecalho, 
                                             largura_col=5000)
                    
                    tb_time = Tabela_excel()
                    tb_time.Criar_tabela(nome_tabela=patch+dataset_name+'-time', 
                                             folhas=['modelos'], 
                                             cabecalho=cabecalho, 
                                             largura_col=5000)
                    
                    
                # running each model
                for xxx, model in enumerate(models):
                    
                    # creating a copy
                    model = copy.deepcopy(model)
                    
                    # to measure the time
                    model.start_time = time.time()
                    
                    # runnning the current model
                    if(cross_validation==True):
                        model.run(labels=labels,
                                  stream=stream_records,
                                  cross_validation=True, 
                                  fold=j, 
                                  qtd_folds=30)
                    else:
                        model.run(labels=labels,
                                  stream=stream_records)
                    
                    # to measure the time
                    model.end_time = time.time()
                    
                    # storing informations
                    self.saveInformation(j, 
                                         xxx,
                                         patch, 
                                         tb_accuracy,
                                         tb_time, 
                                         dataset_name, 
                                         model.NAME, 
                                         model.returnPredictions(), 
                                         model.returnTarget(), 
                                         model.accuracyGeneral(),
                                         model.timeExecution())
                    

def main():
    
    # 1. INSTANTIATE THE EXPERIMENT CLASS
    EXP = Experiment(pasta='AAA')
    
    # 2. INSTANTIATE THE MODELS TO BE RUNNED
    #2.1.
    GMMVRD1 = GMM_VRD()
    
    #2.2.
    DYNSE = Dynse()
    
    #2.3
    IGMMCD = IGMM_CD()
    
    #2.4
    
    #3. DEFINING THE MODELS THAT ARE GOING TO BE EXECUTED
    MODELS = [GMMVRD1, DYNSE, IGMMCD]
    
    #4. RUNNING THE MODELS CHOSEN
    EXP.run(cross_validation=True, models=MODELS, datasets=[0, 9], executions=[0, 1])
            
if __name__ == "__main__":
    main() 
    
