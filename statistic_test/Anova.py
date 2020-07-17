'''
Created on 2 de jul de 2019
@author: gusta
'''

from statistic_test.LerDados import Ler_dados
from sklearn.metrics import accuracy_score
from statsmodels.formula.api import ols
import statsmodels.stats.multicomp
import matplotlib.pyplot as plt
import statsmodels.api as sm
import researchpy as rp
import seaborn as sns
import pandas as pd
import numpy as np

class AnovaTwoWay():
    def __init__(self, pasta, arquivos, execucoes, modelos, labels):
        '''
        :pasta: location of the data
        :arquivos: the name of the achives that will be read
        :execucoes: the number of executions 
        :models: models tested
        
        # example of data format and analysis
        
        # two-way anova example
        acuracia    T        dataset
        0.626918     T = 1    circles
        0.617862     T = 1    circles
        0.626918     T = 2    sine1
        0.617862     T = 2    sine1
        Accuracy ~ C(Parameters)*C(Datasets)
        
        # three-way anova example
        
        acuracia    kmax        T        dataset
        0.626918     kmax = 1    T = 1  circles
        0.617862     kmax = 2    T = 1  circles
        0.626918     kmax = 1    T = 2  sine1
        0.617862     kmax = 2    T = 2  sine1
        Accuracy ~ C(Kmax)*(T)*C(Datasets)
        '''
        
        self.pasta = pasta
        self.arquivos = arquivos
        self.execucoes = execucoes
        self.modelos = modelos
        self.labels = labels
        self.tbt = Ler_dados()
    
    def runThreeWay(self):
        '''
        method to run the threeway anova
        '''
    
        # reading the data frame
        df = self.generateDataFrameThree()
        
        # generating the statistics
        self.generateStatisticsThree(df)
    
    def generateDataFrameThree(self):
        '''
        Method to generate a dataframe with a correct shape to pass to the anova
        '''
        
        flag=True
        # for para ler cada dataset
        for i in range(len(self.arquivos)):
            
            # lendo todo o arquivo de uma vez
            caminho_arquivo = 'E:/Workspace2/GMM-VRD/projects/'+self.pasta+'/'+self.arquivos[i]+'-accuracy.xls'
            labels, acuracias = self.tbt.obter_dados_arquivo(caminho_arquivo, [1, self.execucoes], [0, self.modelos])
        
            # for para escrever para cada parametro
            for j in range(len(labels)):
                
                # for para escrever para cada acuracia
                for k in range(0, self.execucoes-1):
        
                    #if(j == 0 or j == len(labels)-1):
                    if(True):
        
                        # salvando no dataframe final
                        if(flag):
            
                            # to correct the labels
                            nome_parametro1, nome_parametro2 = self.correctLabels2(labels[j], self.arquivos[i])
                            # to create the dataframe
                            dataframe_final = pd.DataFrame({'Accuracy': [acuracias[j][k]], 
                                                            'Kmax': [nome_parametro1],
                                                            'm': [nome_parametro2], 
                                                            'Datasets': [self.arquivos[i]]})
                            flag=False
                            
                        else:
                            
                            # to correct the labels
                            nome_parametro1, nome_parametro2 = self.correctLabels2(labels[j], self.arquivos[i])
                            # to create the dataframe
                            dataframe_final = dataframe_final.append({'Accuracy': acuracias[j][k], 
                                                                      'Kmax': nome_parametro1,
                                                                      'm': nome_parametro2, 
                                                                      'Datasets': self.arquivos[i]},
                                                                      ignore_index=True)
        
                    # printing the execution
                    print(dataframe_final)
            
        return dataframe_final
    
    def generateStatisticsThree(self, df):
        
        # getting the data frame labels
        labels = list(df.columns.values)
        
        # printing the statistics
        print("\n", rp.summary_cont(df[labels[0]]))
        print("\n", rp.summary_cont(df.groupby([labels[1]]))[labels[0]])
        print("\n", rp.summary_cont(df.groupby([labels[2]]))[labels[0]])
        print("\n", rp.summary_cont(df.groupby([labels[1], labels[2]]))[labels[0]])
        print("\n", rp.summary_cont(df.groupby([labels[1], labels[2], labels[3]]))[labels[0]])
        
        
        # Fits the model with the interaction term
        # This will also automatically include the main effects for each factor
        string = labels[0]+ " ~ " + "C(" +labels[1]+ ")*C("+ labels[2] +")*C("+ labels[3] +")"
        print(string)
        model = ols(string, df).fit()
        
        # Seeing if the overall model is significant
        print("\nOverall model F(%.0f,%.0f) = %.3f, p = %.4f" % (model.df_model, model.df_resid, model.fvalue, model.f_pvalue))
        
        # printing the statistics of test
        print("\n", model.summary())
        
        # Creates the ANOVA table
        res = sm.stats.anova_lm(model, typ=2)
        print("\n", res)
        
        # Organizing the ANOVA table
        table = self.anovaTable(res)
        pd.set_option('display.max_columns', 7)
        pd.set_option('display.width', 1000)
        print("\n", table)
        table.to_excel("../projects/anova_results.xls")
        
        # doing the pair to pair comparison
        mc = statsmodels.stats.multicomp.MultiComparison(df[labels[0]], df[labels[1]])
        mc_results = mc.tukeyhsd()
        print("\n", mc_results)
        
        # doing the pair to pair comparison
        mc = statsmodels.stats.multicomp.MultiComparison(df[labels[0]], df[labels[2]])
        mc_results = mc.tukeyhsd()
        print("\n", mc_results)

    def correctLabels2(self, label, arquivo):
        '''
        Method to correct the name of label and archive
        '''
        
        import re
        parametro1, parametro2 = re.findall("\d+", label)
                            
        # final
        return parametro1, parametro2
      
    def run(self):
        '''
        Function to execute the anova
        '''
        
        # reading the data frame
        #df = self.generateDataFrame()
        df = self.generateDataFrame2()
        
        # generating the statistics
        self.generateStatistics(df)
        
        # plotting the interactions
        self.plotAnova(df)
    
    def chooseDataset(self, number, variation):
        if(number==0):
            name = 'circles'
            name_variation = name+'_'+str(variation)
            drifts = [2000, 4000, 6000]
        
        elif(number==1):
            name = 'sine1'
            name_variation = name+'_'+str(variation)
            drifts = [2000, 4000, 6000, 8000]
        
        elif(number==2):
            name = 'sine2'
            name_variation = name+'_'+str(variation)
            drifts = [2000, 4000, 6000, 8000]
        
        elif(number==3):
            name = 'virtual_5changes'
            name_variation = name+'_'+str(variation)
            drifts = [2000, 4000, 6000, 8000]
        
        elif(number==4):
            name = 'virtual_9changes'
            name_variation = name+'_'+str(variation)
            drifts = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000]
        
        elif(number==5):
            name = 'SEA'
            name_variation = name+'_'+str(variation)
            drifts = [2000, 4000, 6000]
        
        elif(number==6):
            name = 'SEARec'
            name_variation = name+'_'+str(variation)
            drifts = [2000, 4000, 6000, 8000, 10000, 12000, 14000]
            
        elif(number==7):
            name = 'noaa'
            name_variation = name+'_'+str(variation)
            drifts = None
        
        elif(number==8):
            name = 'elec'
            name_variation = name+'_'+str(variation)
            drifts = None
            
        elif(number==9):
            name = 'PAKDD'
            name_variation = name+'_'+str(variation)
            drifts = None
        
        return name, name_variation, drifts
    
# options to query        

    def lerDadosAcuracias(self, models, pasta, numberDataset, variation_max):
    
        # salvar todas as acuracias de todos os modelos
        barras = []
        for model in models:
            
            # salvar as acuracias de um modelo
            modelo_acuracias = []
            for i in numberDataset:
                
                # calculating the standard deviation and mean
                accuracy = []
                for variation in range(variation_max):
                    try:
                        _, name_variation, _ = self.chooseDataset(i, variation)
                        # receiving the target and predict
                        predict = np.asarray(pd.read_csv('../projects/'+pasta+'/'+model+'-'+name_variation+'.csv')['predictions'])
                        target = np.asarray(pd.read_csv('../projects/'+pasta+'/'+model+'-'+name_variation+'.csv')['target'])
                        
                        # calculating the mean of accuracy
                        acc = accuracy_score(target, predict)
                        accuracy.append(acc)
                        
                        # print para acompanhar
                        print(model + "-" + name_variation + ": "+ str(acc))
                        
                    except:
                        accuracy.append(np.mean(accuracy))
                    
                # salvando a acuracia do modelo para o atual dataset
                modelo_acuracias.append(accuracy)
            
            # salvando as acuracias dos modelos
            barras.append(modelo_acuracias)
            
        # retornando as barras
        return barras

    def generateDataFrame2(self):
        '''
        Method to generate a dataframe with a correct shape to pass to the anova
        '''
        
        # acuracias
        data = self.lerDadosAcuracias(self.modelos, self.pasta, [i for i in range(len(self.arquivos))], self.execucoes)
            
        flag=True
        # for para ler cada dataset
        for i in range(len(self.arquivos)):
            
            # for para escrever para cada parametro
            for j in range(len(self.labels)):
                
                # for para escrever para cada acuracia
                for k in range(0, self.execucoes):
        
                    # salvando no dataframe final
                    if(flag):
                        # to correct the labels
                        nome_arquivo, nome_parametro = self.correctLabels(self.labels[j], self.arquivos[i])
                        # to create the dataframe
                        dataframe_final = pd.DataFrame({'Accuracy': [data[j][i][k]], 'Parameters': [nome_parametro], 'Datasets': [self.arquivos[i]]})
                        flag=False
                            
                    else:
                        # to correct the labels
                        nome_arquivo, nome_parametro = self.correctLabels(self.labels[j], self.arquivos[i])
                        # adicionando uma nova linha
                        dataframe_final = dataframe_final.append({'Accuracy': data[j][i][k], 'Parameters': nome_parametro, 'Datasets': nome_arquivo}, ignore_index=True)
                
                # printing the execution
                print(dataframe_final)
            
        return dataframe_final
    
    def generateDataFrame(self):
        '''
        Method to generate a dataframe with a correct shape to pass to the anova
        '''
        
        flag=True
        # for para ler cada dataset
        for i in range(len(self.arquivos)):
            
            # lendo todo o arquivo de uma vez
            caminho_arquivo = 'E:/Workspace2/GMM-VRD/projects/'+self.pasta+'/'+self.arquivos[i]+'-accuracy.xls'
            labels, acuracias = self.tbt.obter_dados_arquivo(caminho_arquivo, [1, self.execucoes], [0, self.modelos])
        
            # for para escrever para cada parametro
            for j in range(len(labels)):
                
                # for para escrever para cada acuracia
                for k in range(0, self.execucoes-1):
        
                    #if(j == 0 or j == len(labels)-1):
                    if(True):
        
                        # salvando no dataframe final
                        if(flag):
            
                            # to correct the labels
                            nome_arquivo, nome_parametro = self.correctLabels(labels[j], self.arquivos[i])
                            # to create the dataframe
                            dataframe_final = pd.DataFrame({'Accuracy': [acuracias[j][k]], 'Parameters': [nome_parametro], 'Datasets': [self.arquivos[i]]})
                            flag=False
                            
                        else:
                            
                            # to correct the labels
                            nome_arquivo, nome_parametro = self.correctLabels(labels[j], self.arquivos[i])
                            # adicionando uma nova linha
                            dataframe_final = dataframe_final.append({'Accuracy': acuracias[j][k], 'Parameters': nome_parametro, 'Datasets': nome_arquivo}, ignore_index=True)
        
                    # printing the execution
                    print(dataframe_final)
            
        return dataframe_final
    
    
    # Calculating effect size
    
    def anovaTable(self, aov):
        '''
        Method to organize the table
        '''
        
        aov['mean_sq'] = aov[:]['sum_sq']/aov[:]['df']
        aov['eta_sq'] = aov[:-1]['sum_sq']/sum(aov['sum_sq'])
        aov['omega_sq'] = (aov[:-1]['sum_sq']-(aov[:-1]['df']*aov['mean_sq'][-1]))/(sum(aov['sum_sq'])+aov['mean_sq'][-1])
        cols = ['sum_sq', 'mean_sq', 'df', 'F', 'PR(>F)', 'eta_sq', 'omega_sq']
        aov = aov[cols]
        return aov
    
    def generateStatistics(self, df):
        
        # getting the data frame labels
        labels = list(df.columns.values)
        
        # printing the statistics
        print("\n", rp.summary_cont(df[labels[0]]))
        print("\n", rp.summary_cont(df.groupby([labels[1]]))[labels[0]])
        print("\n", rp.summary_cont(df.groupby([labels[2]]))[labels[0]])
        print("\n", rp.summary_cont(df.groupby([labels[1], labels[2]]))[labels[0]])
        
        # Fits the model with the interaction term
        # This will also automatically include the main effects for each factor
        string = labels[0]+ " ~ " + "C(" +labels[1]+ ")*C("+ labels[2] +")"
        print(string)
        model = ols(string, df).fit()
        
        # Seeing if the overall model is significant
        print("\nOverall model F(%.0f,%.0f) = %.3f, p = %.4f" % (model.df_model, model.df_resid, model.fvalue, model.f_pvalue))
        
        # printing the statistics of test
        print("\n", model.summary())
        
        # Creates the ANOVA table
        res = sm.stats.anova_lm(model, typ=2)
        print("\n", res)
        
        # Organizing the ANOVA table
        table = self.anovaTable(res)
        pd.set_option('display.max_columns', 7)
        pd.set_option('display.width', 1000)
        print("\n", table)
        table.to_excel("teste.xls")
        
        # doing the pair to pair comparison
        mc = statsmodels.stats.multicomp.MultiComparison(df[labels[0]], df[labels[1]])
        mc_results = mc.tukeyhsd()
        print("\n", mc_results)
        
        # doing the pair to pair comparison
        mc = statsmodels.stats.multicomp.MultiComparison(df[labels[0]], df[labels[2]])
        mc_results = mc.tukeyhsd()
        print("\n", mc_results)

    def plotAnova(self, df):
        '''
        Method to plot the anova
        '''
        
        # plotando o anova de duas vias
        g = sns.catplot(x="Datasets", 
                        y="Accuracy", 
                        hue="Parameters",
                        capsize=.2, 
                        height=4.5,
                        aspect=1.1,
                        kind="point",
                        legend_out=False, 
                        data=df)
        g.set_xticklabels(rotation=30)
        g.despine(left=True)
        plt.subplots_adjust(bottom=0.16, left=0.13, top=0.98, right=0.99)
        plt.grid(True)
        #plt.ylim(0.6, 0.9)
        plt.show()
        
    def correctLabels(self, label, arquivo):
        '''
        Method to correct the name of label and archive
        '''
        
        # alterando o nome do dataset
        nome_arquivo = arquivo
        if(arquivo == "virtual_5changes"):
            nome_arquivo = "virtual 5"
        elif(arquivo == "virtual_9changes"):
            nome_arquivo = "virtual 9"
                            
                            
        ################## AGMM ##########################
        # alterando o nome do parametro
        nome_parametro = label
        if(label == "AGMM (50)"):
            nome_parametro = "m = 50"
        elif(label == "AGMM (100)"):
            nome_parametro = "m = 100"
        elif(label == "AGMM (150)"):
            nome_parametro = "m = 150"
        elif(label == "AGMM (200)"):
            nome_parametro = "m = 200"
        elif(label == "AGMM (250)"):
            nome_parametro = "m = 250"
        elif(label == "AGMM (300)"):
            nome_parametro = "m = 300"
        elif(label == "AGMM (400)"):
            nome_parametro = "m = 400"
            
        elif(label == "AGMM (kmax 2)"):
            nome_parametro = "kmax = 2"
        elif(label == "AGMM (kmax 4)"):
            nome_parametro = "kmax = 4"
        elif(label == "AGMM (kmax 6)"):
            nome_parametro = "kmax = 6"
        elif(label == "AGMM (kmax 8)"):
            nome_parametro = "kmax = 8"


        ################## GMM-VRD ##########################
        elif(label == "train_size50-kmax5"):
            nome_parametro = "m = 50"
        elif(label == "train_size100-kmax5"):
            nome_parametro = "m = 100"
        elif(label == "train_size200-kmax5"):
            nome_parametro = "m = 200"
        elif(label == "train_size300-kmax5"):
            nome_parametro = "m = 300"
        elif(label == "train_size400-kmax5"):
            nome_parametro = "m = 400"
                      
        elif(label == "train_size50-kmax1"):
            nome_parametro = "kmax = 2"
        elif(label == "train_size50-kmax3"):
            nome_parametro = "kmax = 4"
        elif(label == "train_size50-kmax5"):
            nome_parametro = "kmax = 6"
        elif(label == "train_size50-kmax7"):
            nome_parametro = "kmax = 8"
        elif(label == "train_size50-kmax9"):
            nome_parametro = "kmax = 10"
            
        
        
        ################## IGMM-CD ##########################            
        elif(label == "IGMM-CD-kmax1"):
            nome_parametro = "T = 1"
        elif(label == "IGMM-CD-kmax3"):
            nome_parametro = "T = 3"
        elif(label == "IGMM-CD-kmax7"):
            nome_parametro = "T = 7"
        elif(label == "IGMM-CD-kmax9"):
            nome_parametro = "T = 9"
        elif(label == "IGMM-CD-kmax13"):
            nome_parametro = "T = 13"
            
        
        ################## Dynse ##########################
        elif(label == "Dynse-train_size50"):
            nome_parametro = "m = 50"
        elif(label == "Dynse-train_size100"):
            nome_parametro = "m = 100"
        elif(label == "Dynse-train_size200"):
            nome_parametro = "m = 200"
        elif(label == "Dynse-train_size300"):
            nome_parametro = "m = 300"
        elif(label == "Dynse-train_size400"):
            nome_parametro = "m = 400"
            
        # final
        return nome_arquivo, nome_parametro
        
def main():

    datasets = ['Circles',
                'Sine1',
                'Sine2',
                'Virtual5',
                'Virtual9',
                'SEA',
                'SEARec']
    
    ########################## experimento do batch ################################ 
    # modelos
    modelos = ["POGMM_VRD22",
               "POGMM_VRD-batch=100",
               "POGMM_VRD-batch=150",
               "POGMM_VRD-Reusing-Gaussians"]
    # legendas
    legendas = ['M=50',
                'M=100',
                'M=150',
                'M=200']
    
    # instanciando anova
    anova = AnovaTwoWay("ICDM(GaussianVirtual)/Batch", datasets, 20, modelos, legendas)
    # plotando o anova
    anova.run()
    

    ########################## experimento do tamanho do pool ################################### 
    # modelos
    modelos = ["POGMM_VRD-P=5",
               "POGMM_VRD-Reusing-Gaussians",
               "POGMM_VRD-P=15",
               "POGMM_VRD-P=20"]
    # legendas
    legendas = ['P=5',
                'P=10',
                'P=15',
                'P=20']
    
    # instanciando anova
    anova = AnovaTwoWay("ICDM(GaussianVirtual)/Pool Length", datasets, 20, modelos, legendas)
    # plotando o anova
    anova.run()
    ##################################################################################
    
    
    ########################## experimento num models ################################ 
    # modelos
    modelos = ["POGMM_VRD2-NUM=1",
               "POGMM_VRD2-NUM=3",
               "POGMM_VRD-Reusing-Gaussians",
               "POGMM_VRD2-NUM=7"]
    # legendas
    legendas = ['N=1',
                'N=3',
                'N=5',
                'N=7']

    # instanciando anova
    anova = AnovaTwoWay("ICDM(GaussianVirtual)/Num Models", datasets, 20, modelos, legendas)
    # plotando o anova
    anova.run()
    ##################################################################################
if __name__ == '__main__':
    main() 


