'''
Created on 23 de nov de 2018
@author: gusta
'''

from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import numpy as np

def plotMultipleBar(parametros, bases, matrix, xlabel, ylabel, title, width=0.9, height=0.5):
    '''
    metodo para plotar as barras
    :parametros: labels
    :matrix: valores respectivos do label
    '''
    
    for i in range(len(bases)):
        if(bases[i] == 'virtual_5changes'):
            bases[i] = 'virtual 5 drifts'
        elif(bases[i] == 'virtual_9changes'):
            bases[i] = 'virtual 9 drifts'    
    
    # estilo
    plt.style.use('seaborn-whitegrid')

    #cria a figura e o eixo
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    
    # quantidade de valores
    n = len(matrix)
    # criando um arranjo de parametros
    Xi = np.arange(len(parametros))
    
    # flor para plotar as barras
    caixas = []
    for i in range(n):
        # calculo para definir o tamanho de cada barra
        form = Xi - width/2. + i/float(n) * width
        ht = matrix[i]
        
        #plotando a barra
        react = ax1.bar(form, height=ht, width=width/float(n), align="edge", edgecolor='black', label=bases[i])
        caixas.append(react)
    
    # definindo a posicao das barras
    plt.yticks(fontsize=20)
    plt.xticks(Xi, parametros, fontsize=20)
    
    # definindo a aproximacao
    plt.axis([Xi[0]-0.6, Xi[-1]+0.6, height, 1])

    #cria o rotula do primeiro eixo
    ax1.set_ylabel(xlabel, fontsize=20) 
    ax1.set_xlabel(ylabel, fontsize=20)
    
    # criando os indices para a reta
    index = range(len(parametros))
    
    # dados da linha
    dados_linha = [np.mean(matrix[:,i]) for i in range(len(matrix[0]))]
    
    # plotando a linha
    plt.plot(index, dados_linha, color='deepskyblue', label="mean")
    print(dados_linha)
    plt.scatter(range(0, len(index)), dados_linha, s=120, color='deepskyblue', zorder=10)
    
    #plt.title(title)
    plt.legend(loc='upper center', 
               bbox_to_anchor=(0.5, 1.2),
               handletextpad=0.2, 
               labelspacing=0.1,
               columnspacing=0.4,
               ncol=4, 
               fontsize=15, 
               fancybox=True, 
               shadow=True)
    plt.subplots_adjust(bottom=0.14)
    plt.show()


# label dos dados

def plotExemploMutipleBar():
    parametros = ['1','2','3','4','5']
    bases = ['A','B','C','D','E','F','G']
    
    # padroes dos dados
    matrix = np.asarray([[6.1, 7.4, 2.0, 8.5, 4.0],
                          [6.1, 7.4, 2.0, 8.5, 4.0],
                          [8.1, 6.4, 3.0, 6.5, 4.0],
                          [9.1, 5.4, 4.0, 7.5, 4.0],
                          [8.6, 6.3, 5.5, 3.5, 4.0],
                          [8.6, 6.3, 5.5, 3.5, 4.0],
                          [6.1, 7.4, 2.0, 8.5, 4.0]])
    
    # plotando o metodo
    plotMultipleBar(parametros, bases, matrix, 'accuracy', 'media')

def chooseDataset(number, variation):
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

def lerDados(models, pasta, numberDataset, variation_max):
    
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
                    _, name_variation, _ = chooseDataset(i, variation)
                    # receiving the target and predict
                    predict = np.asarray(pd.read_csv('../projects/'+pasta+'/'+model+'-'+name_variation+'.csv')['predictions'])
                    target = np.asarray(pd.read_csv('../projects/'+pasta+'/'+model+'-'+name_variation+'.csv')['target'])
                    
                    # calculating the mean of accuracy
                    acc = accuracy_score(target, predict)
                    accuracy.append(acc)
                    
                    # print para acompanhar
                    print(model + "-" + name_variation + ": "+ str(acc))
                    
                except:
                    continue
                
            # salvando a acuracia do modelo para o atual dataset
            modelo_acuracias.append(np.mean(accuracy))
            #modelo_acuracias.append(accuracy)
        
        # salvando as acuracias dos modelos
        barras.append(modelo_acuracias)
        
    # retornando as barras
    return barras

def lerDadosAcuracias(models, pasta, numberDataset, variation_max):
    
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
                    _, name_variation, _ = chooseDataset(i, variation)
                    # receiving the target and predict
                    predict = np.asarray(pd.read_csv('../projects/'+pasta+'/'+model+'-'+name_variation+'.csv')['predictions'])
                    target = np.asarray(pd.read_csv('../projects/'+pasta+'/'+model+'-'+name_variation+'.csv')['target'])
                    
                    # calculating the mean of accuracy
                    acc = accuracy_score(target, predict)
                    accuracy.append(acc)
                    
                    # print para acompanhar
                    print(model + "-" + name_variation + ": "+ str(acc))
                    
                except:
                    continue
                
            # salvando a acuracia do modelo para o atual dataset
            modelo_acuracias.append(accuracy)
        
        # salvando as acuracias dos modelos
        barras.append(modelo_acuracias)
        
    # retornando as barras
    return barras

def plotStackedBar4(barras, datasets, legendas):
    
    # Names of group and bar width
    barWidth = 0.5
    
    # The position of the bars on the x-axis
    r = [i for i in range(len(datasets))]
     
    # colors for the box
    colors = cm.rainbow(np.linspace(0, 1, len(barras)))
     
    # criando as barras organizadas
    for x, ha, hb, hc, hd in zip(r, barras[0], barras[1], barras[2], barras[3]):
        if(x==0):
            for i, (h, c) in enumerate(sorted(zip([ha, hb, hc, hd], colors))):
                plt.bar(x, h, color=c, edgecolor='black', label=legendas[i], zorder=-i)
        else:
            for i, (h, c) in enumerate(sorted(zip([ha, hb, hc, hd], colors))):
                plt.bar(x, h, color=c, edgecolor='black', zorder=-i)

    # ajustando a imagem
    #ajuste = barras[1][0]-barras[0][0]
    ajuste = 0.01
    
    # alterando a legenda
    datasets[3] = "Virtual5"
    datasets[4] = "Virtual9"
    
    # Custom X axis
    plt.axis([r[0]-barWidth, r[-1]+barWidth, np.min(barras)-ajuste, np.max(barras)+ajuste])
    plt.xticks(r, datasets)
    plt.xlabel("Datasets")
    plt.ylabel("Accuracy") 
    # Show graphic
    plt.legend()
    plt.show()

def plotStackedBar3(barras, datasets, legendas):
    
    # Names of group and bar width
    barWidth = 0.5
    
    # The position of the bars on the x-axis
    r = [i for i in range(len(datasets))]
     
    # colors for the box
    colors = cm.rainbow(np.linspace(0, 1, len(barras)))
     
    # criando as barras organizadas
    for x, ha, hb, hc in zip(r, barras[0], barras[1], barras[2]):
        if(x==0):
            for i, (h, c) in enumerate(sorted(zip([ha, hb, hc], colors))):
                plt.bar(x, h, color=c, edgecolor='black', label=legendas[i], zorder=-i)
        else:
            for i, (h, c) in enumerate(sorted(zip([ha, hb, hc], colors))):
                plt.bar(x, h, color=c, edgecolor='black', zorder=-i)

    # ajustando a imagem
    #ajuste = barras[1][0]-barras[0][0]
    ajuste = 0.01
    
    # alterando a legenda
    datasets[3] = "Virtual5"
    datasets[4] = "Virtual9"
    
    # Custom X axis
    plt.axis([r[0]-barWidth, r[-1]+barWidth, np.min(barras)-ajuste, np.max(barras)+ajuste])
    plt.xticks(r, datasets)
    plt.xlabel("Datasets")
    plt.ylabel("Accuracy") 
    # Show graphic
    plt.legend()
    plt.show()
    
def plotBoxplot(barras, legendas):
    plt.boxplot(barras)
    plt.xticks([i for i in range(1, len(barras)+1)], legendas)
    plt.show()

def multipleBoxplot(datasets, legendas, dados):
    '''
    metodo para gerar um boxplot para as duas series contendo o desempenho de todos os classificadores ao longo da variacao da porcentagem de treinamento
    :param: dataset: nome do dataset a ser plotado
    :param: metrica: nome da metrica a ser plotada
    :return: retorna o plot correspondente aos parametros consultados
    '''
    
    data_box = []
    # labels de cada quadro
    for i, x_label in enumerate(datasets):
        quadro = x_label
        
        acuracias = []
        # pegando a acuracia de cada algoritmo
        for alg in dados:
            acuracias.append(alg[i]) 
        
        # adicionando os resultados para cada quadro
        data_box.append([quadro, acuracias])
    
    # colors for the box
    colors = ['pink', 'lightblue', 'lightgreen', 'gray']
    
    # instantiating each square of boxplot  
    fig, axes = plt.subplots(ncols=len(data_box), sharey=True)
    fig.subplots_adjust(wspace=0)
    
    # atribuindo o label do eixo de y
    axes[0].set_ylabel('Accuracy')
    
    # creating a figure with each boxplot instanted in the previous line
    for ax, d in zip(axes.flatten(), data_box):  
        # creating the current boxplot with its respective label   
        box = ax.boxplot([d[1][legendas.index(attempt)] for attempt in legendas], patch_artist=True)
        # for each square and boxplot define the current label
        #ax.set_xticklabels(attemptlist, rotation=90, ha='right')
        ax.set(xlabel=d[0])
        # defining the collor for each boxplot
        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)
        # creating a grid
        ax.yaxis.grid(True, alpha=0.1)
        
    # plotting the legend
    axes.flatten()[-2].legend(handles=box['boxes'],
                              labels=legendas, 
                              loc='upper center', 
                              bbox_to_anchor=(0, 0.93, -3, 0.2), 
                              ncol=len(legendas))
    
    fig.set_size_inches(5.8, 3.8, forward=True)
    plt.subplots_adjust(bottom=0.16, left=0.13, top=0.92, right=0.99)
    plt.show()

def main():
    
    #############################################################################################
    # datasets
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
    # acuracias
    data = lerDadosAcuracias(modelos, "ICDM(GaussianVirtual)/Batch", [0,1,2,3,4,5,6], 20)
    # plotando as barras
    multipleBoxplot(datasets, legendas, data)
    ##################################################################################


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
    # acuracias
    data = lerDadosAcuracias(modelos, "ICDM(GaussianVirtual)/Pool Length", [0,1,2,3,4,5,6], 20)
    # plotando o boxplot
    multipleBoxplot(datasets, legendas, data)
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
    # acuracias
    data = lerDadosAcuracias(modelos, "ICDM(GaussianVirtual)/Num Models", [0,1,2,3,4,5,6], 20)
    # plotando o boxplot
    multipleBoxplot(datasets, legendas, data)
    ##################################################################################

    
if __name__ == '__main__':
    main()
    


