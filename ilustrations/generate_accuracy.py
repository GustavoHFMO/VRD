'''
Created on 6 de set de 2018
@author: gusta
'''
from data_streams.adjust_labels import Adjust_labels
al = Adjust_labels()
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
#print(plt.style.available)

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

def calculateLongAccuracy(target, predict, batch):
    '''
    method to calculate the model accuracy a long time
    :param: target:
    :param: predict:
    :param: batch:
    :return: time series with the accuracy 
    '''
        
    time_series = []
    for i in range(len(target)):
        if(i % batch == 0):
            time_series.append(accuracy_score(target[i:i+batch], predict[i:i+batch]))
                               
    return time_series

def cross_validation(predict, target, qtd_folds, fold):
    
    def leave_out(qtd_folds, fold, count):
        '''
        Method to use the cross validation to data streams
        '''
        
        # if the current point reach the maximum, then is reseted 
        if(count == qtd_folds):
            count = 0

        # false if the fold is equal to count
        if(count == fold):
            Flag = True
        else:
            Flag = False
        
        # each iteration is accumuled an point
        count += 1
        
        #returning the flag
        return Flag, count
    
    
    # getting the points to be excluded
    
    indices = []
    count = 0
    for i in range(len(target)):
        flag, count = leave_out(qtd_folds, fold, count)
        if(flag):
            indices.append(i)
            

    # getting the data with cross validation
    new_predict = []
    new_target = []
    for i in range(len(predict)):
        if(i not in indices):
            new_predict.append(predict[i])
            new_target.append(target[i])
    
    # returning the data    
    return new_predict, new_target 
    
def plotStreamAccuracyStandardDeviation(models, pasta, numberDataset, variation_max, batch):
    '''
    method to create different types of graph of accuracy for several models
    '''
    
    plt.style.use('seaborn-white')
    
    for x, model in enumerate(models):
        
        # calculating the standard deviation and mean
        final_time_series = []    
        accuracy = []
        for variation in range(variation_max):
                
            name, name_variation, drifts = chooseDataset(numberDataset, variation)
            # receiving the target and predict
            predict = np.asarray(pd.read_csv('../projects/'+pasta+'/'+model+'-'+name_variation+'.csv')['predictions'])
            target = np.asarray(pd.read_csv('../projects/'+pasta+'/'+model+'-'+name_variation+'.csv')['target'])
            
            # calculating the mean of accuracy
            accuracy.append(accuracy_score(target, predict))
            
            # calculating the accuracy a long time
            time_series = calculateLongAccuracy(target, predict, batch)
            
            # final time series append
            final_time_series.append(time_series)
            
        
        time_series_mean = np.mean(final_time_series, axis=0)
        time_series_std = np.std(final_time_series, axis=0)
        time_series_upper = time_series_mean + time_series_std
        time_series_lower = time_series_mean - time_series_std
        
        # plotting 
        text = model +": %.3f (%.3f)" % (np.mean(accuracy), np.std(accuracy))
        line = plt.plot(time_series_mean, label=text)
        plt.plot(time_series_upper, ':', alpha=0.4, color=line[0].get_color())
        plt.plot(time_series_lower, '-.', alpha=0.4, color=line[0].get_color())
    
    plt.title("dataset: "+name)
    plt.ylabel('Accuracy')
    plt.xlabel('Batches')
    
    # plotting the legend
    plt.plot(time_series_mean[:1], ':', label="upper std", color='gray')
    plt.plot(time_series_mean[:1], '-.', label="lower std", color='gray')
            
    # plotting drifts
    for x, i in enumerate(drifts):
        i = i/batch
        plt.axvline(i, linestyle='dashed', color = 'black', label='changes', alpha=0.1)
        if(x==0):
            plt.legend(loc='lower left')
    plt.show()
    
def plotStreamAccuracyStandardDeviationFill(models, pasta, numberDataset, variation_max, batch, drifts_plot, metric="accuracy"):
    '''
    method to create different types of graph of accuracy for several models
    '''
    
    plt.style.use('seaborn-white')
    
    for x, model in enumerate(models):
        
        # calculating the standard deviation and mean
        final_time_series = []    
        accuracy = []
        for variation in range(variation_max):
                
            try:
                _, name_variation, drifts = chooseDataset(numberDataset, variation)
                #print(name_variation)
                # receiving the target and predict
                predict = np.asarray(pd.read_csv('../projects/'+pasta+'/'+model+'-'+name_variation+'.csv')['predictions'])
                target = np.asarray(pd.read_csv('../projects/'+pasta+'/'+model+'-'+name_variation+'.csv')['target'])
                
                modelos_cross = ["POGMM_VRD22", "LDD-DSDA", "POGMM_VRD-Nothing", "POGMM_VRD-Pool-training", "POGMM_VRD-Reusing-Gaussians"]
                if(drifts_plot==True and model not in modelos_cross):
                    predict, target = cross_validation(predict, target, 20, variation)
                    
                elif(drifts_plot==False and model == "Proposed Method"):
                    predict, target = cross_validation(predict, target, 20, variation)
                
                # calculating the mean of accuracy
                if(metric=="accuracy"):
                    acc = accuracy_score(target, predict)
                elif(metric=="gmean"):
                    acc = geometric_mean_score(target, predict)
                
                # printing and adding the performance
                print(acc)
                accuracy.append(acc)
                
                # calculating the accuracy a long time
                time_series = calculateLongAccuracy(predict, target, batch)
                
                # final time series append
                final_time_series.append(np.asarray(time_series))
            except:
                continue
            
        print("----")
        
        # to correct the batches
        tam = len(final_time_series[0])
        for i in range(1, len(final_time_series)):
            if(len(final_time_series[i]) != tam):
                final_time_series[i] = final_time_series[i][:tam]

        
        # plotting the aot
        time_series_mean = np.mean(final_time_series, axis=0)
        time_series_std = np.std(final_time_series, axis=0)
        time_series_upper = time_series_mean + time_series_std
        time_series_lower = time_series_mean - time_series_std

                    
        # plotting
        model = correctLabels(model) 
        text = model +": %.3f (%.3f)" % (np.mean(accuracy), np.std(accuracy))
        
        line = plt.plot(time_series_mean, label=text)
        plt.fill_between(np.arange(len(time_series_mean)), 
                         time_series_upper, 
                         time_series_lower, 
                         color=line[0].get_color(),
                         edgecolor=line[0].get_color(),
                         linewidth=0.0,
                         alpha=0.1)
    
    letras = 22
    plt.yticks(fontsize=letras)
    plt.xticks(fontsize=letras)
    
    #plt.title("dataset: "+name)
    plt.ylabel('Accuracy', fontsize=letras)
    plt.xlabel('Batches', fontsize=letras)
    
    plt.subplots_adjust(bottom=0.13, top=0.98, left=0.16, right=0.99)
    
    if(drifts_plot):
        # plotting drifts
        for x, i in enumerate(drifts):
            i = (i/batch)-2
            plt.axvline(i, linestyle='dashed', color = 'black', label='Concept Drift', alpha=0.1)
            if(x==0):
                plt.legend(loc='lower left',
                   fontsize=15)
    else:
        plt.legend(loc='lower left',
                   fontsize=15)
    plt.show()

def correctLabels(model):
    if(model == 'train_size50-kmax1'):
        model = 'GMM-VRD'
    elif(model == 'Proposed Method'):
        model = 'GMM-VRD'
    elif(model == 'AGMM'):
        model = 'OGMMF-VRD'
    elif(model == 'Completo'):
        model = 'Complete'
    elif(model == 'Dynse-priori'):
        model = 'Dynse'
    elif(model == 'Dynse-priori-age'):
        model = 'Dynse'
    elif(model == 'POGMM_VRD22'):
        model = 'POGMM-VRD'
    
        
    elif(model == 'POGMM_VRD-Nothing'):
        model = 'W/0 Pool Training'
    elif(model == 'POGMM_VRD-Pool-training'):
        #model = 'With Pool Training'
        model = 'W/0 Gaussian Reuse'
    elif(model == 'POGMM_VRD-Reusing-Gaussians'):
        model = 'With Gaussian Reuse'
        
    return model
    
def main():
    
    # mechanisms analysis
    #models = ['POGMM_VRD-Nothing', 'POGMM_VRD-Pool-training']
    #plotStreamAccuracyStandardDeviationFill(models, 'ICDM(GaussianVirtual)/Mechanisms', 4, 20, 250, True)
    
    #models = ['POGMM_VRD-Pool-training', 'POGMM_VRD-Reusing-Gaussians']
    #plotStreamAccuracyStandardDeviationFill(models, 'ICDM(GaussianVirtual)/Mechanisms', 1, 20, 250, True)
    
    # synthetic
    #models = ['POGMM_VRD22', 'LDD-DSDA', 'train_size50-kmax1', 'IGMM-CD', 'Dynse-priori-age']#'Dynse-priori-age or Dynse-priori'
    #plotStreamAccuracyStandardDeviationFill(models, 'ICDM(GaussianVirtual)/Artificial', 6, 20, 250, True, "accuracy")
    
    # real
    models = ['POGMM_VRD22', 'LDD-DSDA', 'Proposed Method', 'IGMM-CD', 'Dynse-priori']
    plotStreamAccuracyStandardDeviationFill(models, 'ICDM(GaussianVirtual)/Real', 9, 20, 250, False, "accuracy")
    #===========================================================================
    
if __name__ == "__main__":
    main()    



