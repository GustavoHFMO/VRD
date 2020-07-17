#-*- coding: utf-8 -*-
'''
Created on 10 de nov de 2018
@author: gusta
'''

from statistic_test.LerDados import Ler_dados
from sklearn.metrics import accuracy_score
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import copy

class WilcoxonTest():
    def __init__(self, labels, data, p_value=0.05):
        self.labels = labels
        self.data = data
        self.p_value = p_value

    def do(self, reverse=False):
        #computando o wilcoxon
        _, p_value = wilcoxon(self.data[0] - self.data[1])

        # computando a media
        mean0 = np.mean(self.data[0])
        mean1 = np.mean(self.data[1])
        
        # labels para print        
        label0 = self.labels[0]
        label1 = self.labels[1]
                    
        # flag para retorno
        flag = False
        
        # analisando os dados
        if(p_value < self.p_value):
            if(mean0 > mean1):
                label0 = self.labels[1]
                label1 = self.labels[0]
                mean0 = np.mean(self.data[1])
                mean1 = np.mean(self.data[0])
                
            print("the algorithm: "+label0+" ("+str(mean0)+") is the less than: "+label1+" ("+str(mean1)+")")
            flag = True
            
        else:
            print("the algorithms are statistically equal!")
            print("the algorithm: "+label0+": "+str(mean0)+" \n"+label1+": "+str(mean1))
        
        
        # printando final
        print("paired wilcoxon-test p-value: ", p_value)
        
        # retornando o resultado        
        return flag, p_value
        
    def Exemplo_executavel(self):
        #acuracias dos modelos, cada coluna é um modelo
        data1 = np.asarray([3.88, 5.64, 5.76, 4.25, 5.91, 4.33])
        data2 = np.asarray([30.58, 30.14, 16.92, 23.19, 26.74, 10.91])
                            
        #label dos modelos, cada coluna é um modelo
        names_alg = ["alg1", 
                     "alg2"]
        
        wc = WilcoxonTest(names_alg, [data1, data2])
        wc.do()

class PlotWilcoxonTest():
    def __init__(self):
        pass
    
    def gerarAcuracias(self, model, pasta, dataset, variation_max):
        
        # acuracia para cada dataset
        acuracias = []
            
        # for para cada execucao do dataset
        for variation in range(variation_max):
                        
            try:
                        
                # receiving the target and predict
                predict = np.asarray(pd.read_csv('../projects/'+pasta+'/'+model+'-'+dataset+'_'+str(variation)+'.csv')['predictions'])
                target = np.asarray(pd.read_csv('../projects/'+pasta+'/'+model+'-'+dataset+'_'+str(variation)+'.csv')['target'])
    
                # calculando a acuracia
                acc = accuracy_score(target, predict)
                
                # salvando as acuracias para cada dataset
                acuracias.append(acc)
                        
                # print para acompanhar
                print(model + "-" + dataset + "_" + str(variation) + ": "+ str(acc))
                                            
            except:
                #acuracias.append(acuracias[0])
                acuracias.append(np.mean(acuracias))
                
        # retornando a matriz de modelos
        return np.asarray(acuracias)
    
    def correctNames(self, arquivos):
        '''
        Method to correct the names of the datasets
        '''
        
        novos = copy.deepcopy(arquivos)
        for i in range(len(novos)):
            if(novos[i] == "virtual_5changes"):
                novos[i] = "virtual5"
            elif(novos[i] == "virtual_9changes"):
                novos[i] = "virtual9"
                
        return novos
                
    def comparacaoPlot(self, arquivos, pasta, comparacao1, comparacao2, titulo, execucoes=30, ajuste_pvalor=0.3):
        
        # adicionando o estilo
        plt.style.use('seaborn-white')
        
        # variavel para salvar desempenhos
        desempenhos = []
        
        # variavel para salvar os niveis de diferenca
        diferencas = [''] * len(arquivos)
        
        # para ler cada arquivo
        for i in range(len(arquivos)):
            data1 = self.gerarAcuracias(comparacao1, pasta, arquivos[i], execucoes)
            data2 = self.gerarAcuracias(comparacao2, pasta, arquivos[i], execucoes)
            
            # calculando a diferenca entre os desempenhos
            desempenhos.append(np.mean(data2)-np.mean(data1))
        
            # computando o wilcoxon para a comparacao de 5%
            wc = WilcoxonTest([comparacao1, comparacao2], [data1, data2], p_value=0.05)
            _, p_value = wc.do(reverse=True)
            
            # salvando na variavel se e diferente para 5%
            diferencas[i] = p_value
                
        # alterando labels
        arq = self.correctNames(arquivos)
        
        #plotando a diferenca
        y_pos = np.arange(len(arq)) 
        rect = plt.bar(arq, desempenhos, align='center', alpha=0.5, edgecolor="black")
        # letras tamanho
        letras = 13
        # adicionando os pvalues
        self.marker2(rect, diferencas, letras, ajuste_pvalor)
        
        # plotando o grafico
        plt.xticks(y_pos, arq, rotation=45, size=letras)
        plt.yticks(size=letras)
        plt.ylabel('Accuracy Difference', size=letras)
        plt.title(titulo)
        plt.subplots_adjust(bottom=0.2, top=0.95, left=0.22, right=0.99)
        plt.legend()
        plt.show()
    
    def marker2(self, rects, labels, letras, ajuste_pvalor):
        """Attach a text label above each bar in *rects*, displaying its height."""
        
        # for to plot for each bar
        for i, rect in enumerate(rects):
            
            if(labels[i] < 0.05):
                labels[i] = "%.1E*" % labels[i]
            else:
                labels[i] = "%.1E" % labels[i]
                
            height = rect.get_height()
            plt.text(rect.get_x()-rect.get_width()/25+ajuste_pvalor, height, labels[i], color='black', size=letras)
        
    def marker(self, rects, labels):
        """Attach a text label above each bar in *rects*, displaying its height."""
        
        # for helping plot just one time the legend
        aux = [False, False]
        
        # for to plot for each bar
        for i, rect in enumerate(rects):
            
            # adding the label and marker
            if(labels[i]=='*'):
                mark = 'v'
                lb = 'alpha = 5%'
                j = 0
            elif(labels[i]=='**'):
                mark = 'd'
                lb = 'alpha = 10%'
                j = 1
            
            # to plot the legend one time
            if(aux[j] == False):
                height = rect.get_height()
                plt.plot(rect.get_x()+rect.get_width()/2, height, mark, color='black', label=lb)
                aux[j] = True
            else:
                height = rect.get_height()
                plt.plot(rect.get_x()+rect.get_width()/2, height, mark, color='black')
        
def main():
    
    arquivos = ['circles',
                'sine1',
                'sine2',
                'virtual_5changes',
                'virtual_9changes',
                'SEA',
                'SEARec']
    
    pasta = "ICDM(GaussianVirtual)/Mechanisms"
    
    ex = 20
    ajuste = -0.06

    wc = PlotWilcoxonTest()
    wc.comparacaoPlot(arquivos, 
                      pasta, 
                      'POGMM_VRD-Nothing', 
                      'POGMM_VRD-Pool-training', 
                      'Pool Training',
                      execucoes=ex,
                      ajuste_pvalor=ajuste)
    
    
    wc = PlotWilcoxonTest()
    wc.comparacaoPlot(arquivos, 
                      pasta, 
                      'POGMM_VRD-Pool-training', 
                      'POGMM_VRD-Reusing-Gaussians', 
                      'Pool Reusing',
                      execucoes=ex,
                      ajuste_pvalor=ajuste)
    
    
if __name__ == '__main__':
    main()    
        