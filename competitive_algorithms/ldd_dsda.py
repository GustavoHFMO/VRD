'''
Created on 6 de jul de 2020
@author: gusta
LIU, Anjin et al. Regional concept drift detection and density synchronized drift adaptation. 
In: IJCAI International Joint Conference on Artificial Intelligence. 2017.
'''


# Importing dynamic selection techniques:
from data_streams.adjust_labels import Adjust_labels
from competitive_algorithms.prequential_super import PREQUENTIAL_SUPER
al = Adjust_labels()
from streams.readers.arff_reader import ARFFReader
from sklearn.neighbors import NearestNeighbors
from sklearn.naive_bayes import GaussianNB
from sklearn.utils import resample
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
from scipy.stats import norm
import numpy as np

class LDD_DIS:
    def __init__(self, rho=0.1, alpha=0.05):
        '''
        Method to detect regional drifts
        :p: neighbourhood size
        :a: drift significance level
        '''
        
        self.rho = rho
        self.alpha = alpha
        
    def belong(self, x, Set):
        '''
        Method to verify is an array element is in another array
        :x: array element
        :Set: array element
        '''

        # return true or false        
        return (x==Set).all(1).any() 
        
    def intersection(self, d, knn, D_knn, D_query):
        '''
        Method to calculate the intersection of two sets
        :d: the instance used to calculate the neighborhood
        :knn: the knn instance
        :D_knn: the set used with knn
        :D_query: the set calculate the insersection
        '''
        
        # getting the indices of the neighbors of d 
        _, indices = knn.kneighbors([d])
        # getting the neighbors of d  
        neighborhood = [D_knn[i] for i in indices][0]
                

        # verifying the intersection of the neighbors in D1_
        cont = 0
        for d in neighborhood:
            if(self.belong(d, D_query)):
                cont += 1
             
        # returning the elements intersected           
        return cont
    
    def LDD(self, d, knn, D_knn, D1_, D2_):
        '''
        Method to calculate the intersection of two sets
        :d: the instance used to calculate the neighborhood
        :knn: the knn instance
        :D_knn: the set used with knn
        :D1_: the initial batch
        :D2_: the current batch
        '''

        # calculating the intersection for both sets
        D1_intersection = self.intersection(d, knn, D_knn, D1_)
        D2_intersection = self.intersection(d, knn, D_knn, D2_)
        
        # to avoid errors
        if(D2_intersection == 0): D2_intersection = 0.001 
        
        # computing the LDD for the data consulted
        delta = (D1_intersection/D2_intersection)-1
        
        # returning the current delta
        return delta  
        
    def detect(self, D1, D2):
        '''
        Drift Instance Selection method to detect changes in the data
        :D1: initial batch of instances
        :D2: current batch of instances
        '''
        
        
        ########## 1. Estimate the Distribution of LDD #######################################################

        # merging the two batches (Line 1)
        D = np.vstack((D1, D2))
        
        # k to be used in the KNN
        k = int(len(D) * self.rho)
        
        # computing the KNN for the data (Lines 2 to 4)
        knn = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(D)
        
        
        # shuffling the merged data (Line 5) 
        np.random.shuffle(D)
        # resampling new two batches (Line 5)
        D1_, D2_ = D[0:len(D1)], D[len(D1):]

        
        # variable to store the ldd to all instances
        Deltas = []
        # computing the LDD for each instance in D (Lines 6 to 11)
        for d in D:
            # verifying if d is in D1_ (Line 7)
            if(self.belong(d, D1_)):
                # computing the ldd for d (Line 8) 
                Deltas.append(self.LDD(d, knn, D, D2_, D1_)) 
            # if d ins not D1_, then d is in D2_ (Lines 9 to 10)
            else:
                # computing the ldd for d (Line 10)
                Deltas.append(self.LDD(d, knn, D, D1_, D2_))
        
        
        # computing the normal cumulative distribution function (Line 12)
        sigma_dec = norm.ppf(self.alpha, loc=0, scale=np.std(Deltas))
        # computing the normal cumulative distribution function (Line 13)
        sigma_inc = norm.ppf((1-self.alpha), loc=0, scale=np.std(Deltas))
        
        ########## 1. ########################################################################################
        
        
        ########## 2. Creation of sets #######################################################################
        # creating the variables for each new sebset
        D1_dec = []
        D1_sta = []
        D1_inc = []
        #
        D2_dec = []
        D2_sta = []
        D2_inc = []
        
        
        # for to calculate the new sets (Line 14)
        for d in D:
            # verifying if the instance belong to D1 (Line 15)
            if(self.belong(d, D1)):
                
                # computing the LDD to the original sets (Line 16) 
                delta = self.LDD(d, knn, D, D2, D1)
                
                # comparison of the current delta to the estimated (Line 17)
                if(delta < sigma_dec):
                    # (Line 18)
                    D1_dec.append(d)
                # comparison of the current delta to the estimated (Line 19)
                elif(delta > sigma_inc):
                    # (Line 19)
                    D1_inc.append(d)
                # if the instance do not belong to anyone then it is stable (Line 21)
                else:
                    # (Line 20)
                    D1_sta.append(d)
            
            
            # if the data belongs to D2 (Line 23)
            else:
                
                # computing the LDD to the original sets (Line 16) 
                delta = self.LDD(d, knn, D, D1, D2)
                    
                # comparison of the current delta to the estimated (Line 17)
                if(delta < sigma_dec):
                    # (Line 18)
                    D2_dec.append(d)
                # comparison of the current delta to the estimated (Line 19)
                elif(delta > sigma_inc):
                    # (Line 19)
                    D2_inc.append(d)
                # if the instance do not belong to anyone then it is stable (Line 21)
                else:
                    # (Line 20)
                    D2_sta.append(d)
        
        
        ########## 2. ########################################################################################
        
        # returning the new subsets
        return np.asarray(D1_dec), np.asarray(D1_sta), np.asarray(D1_inc), np.asarray(D2_dec), np.asarray(D2_sta), np.asarray(D2_inc)
    
class LDD_DSDA(PREQUENTIAL_SUPER):
    def __init__(self, train_size=100):
        '''
        a regional drift adaptation algorithm to synchronize the density discrepancies based on the identified drifted instances.
        :W: the batch to train the model and initialize the drift detector
        '''
        
        # the size of W the window to train the model
        self.TRAIN_SIZE = train_size
        # its drift detector
        self.drift_detector = LDD_DIS()

        # method name
        self.NAME = "LDD-DSDA"
                
        # auxiliar variable
        self.PREDICTIONS = []
        self.TARGET = []
        self.count = 0
        
    def buildLearner(self, train):
        '''
        Method to train the base learner
        :train: the window with the patterns
        '''
        
        # splitting the data in patterns and labels
        X, Y = train[:, 0:-1], train[:,-1]
        
        # creating the classifier
        learner = GaussianNB()
        
        # fitting the classifier on the data
        learner.fit(X, Y)
        
        # returning the classifier trained
        return learner
    
    def incrementWindow(self, W, x):
        '''
        method to icrement the window under the example
        :param: W: window that will be updated 
        :param: x: example that will be inserted 
        '''
        
        aux = [None] * (len(W)+1)
        aux[:-1] = W
        aux[-1] = x
        
        return np.asarray(aux)
    
    def mergeData(self, D_train_dec, D_train_sta, D_train_inc, D_buffer_dec, D_buffer_sta, D_buffer_inc, Train, Buffer):
        '''
        The intuitive idea of LDD-DSDA is to merge existing data
        Dtrain with the recently buffered data buffer and resample a
        new batch of training data Train so that the new Train has
        the same distribution as Bufffer
        '''
        
        def stack_(a, b):
            '''
            Simple method to avoid mistakes
            '''
            
            if(len(a) == 0):
                return b
            elif(len(b) == 0):
                return a 
            else:
                return np.vstack((a, b))
        
        def resample_(a, size):
            '''
            method to resample without mistake
            '''
            
            if(len(a) == 0 or size == 0):
                return []
            else:
                return resample(a, n_samples=size)
            
        # first merging rule
        if(len(D_train_dec)/len(Train) == len(D_buffer_inc)/len(Buffer)):
            return Buffer
        
        # second merging rule
        else:
            D_inc = stack_(D_train_inc, D_buffer_dec)
            D_sta = stack_(D_train_sta, D_buffer_sta)
            D_dec = stack_(D_train_dec, D_buffer_inc)
    
            # resampling the data
            D_sample_inc = resample_(D_inc, len(D_buffer_inc))
            D_sample_sta = resample_(D_sta, len(D_buffer_sta))
            D_sample_dec = resample_(D_dec, len(D_buffer_dec))
            
            # final data to train the model
            Train = stack_(D_sample_inc, D_sample_sta)
            Train = stack_(Train, D_sample_dec)
            
            # returning the new training batch
            return Train
    
    def plotLearnedBoundaries(self, model, X, Y, titulo, show=True):
        
        def plotData(x, y):
            # defining the colors of each gaussian
            unique, _ = np.unique(y, return_counts=True)
            colors = plt.cm.rainbow(np.linspace(0, 1, len(unique)))
            marks = ["^", "o", '+', ',']
            
            # creating the image
            plt.subplot(111)
                
            # receiving each observation per class
            classes = []
            for i in unique:
                aux = []
                for j in range(len(y)):
                    if(y[j] == i):
                        aux.append(x[j])
                classes.append(np.asarray(aux))
            classes = np.asarray(classes)
            
            # plotting each class
            for i in unique:
                i = int(i)
                plt.scatter(classes[i][:,0],
                            classes[i][:,1],
                            color = colors[i],
                            marker = marks[i])
                            #label = 'class '+str(i))
            
        def make_meshgrid(x, y, h=.01):
            x_min, x_max = x.min() - 1, x.max() + 1
            y_min, y_max = y.min() - 1, y.max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
            return xx, yy
        
        def plot_contours(ax, clf, xx, yy, **params):
            Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            ax.contourf(xx, yy, Z, **params)
            ax.contour(xx, yy, Z,
                    linewidths=3, 
                    linestyles='solid',
                    cmap=plt.cm.rainbow, 
                    zorder=0)
            # adding the label of true boundaries
            c = plt.cm.rainbow(np.linspace(0, 1, 3))
            ax.plot([],[], linewidth=2, linestyle='solid', color=c[0], label="Learned Boundaries")
            
        # plot data
        plotData(X, Y)
        # creating the boundaries
        X0, X1 = X[:, 0], X[:, 1]
        xx, yy = make_meshgrid(X0, X1)
        plot_contours(plt, model, xx, yy, cmap=plt.cm.rainbow, alpha=0.1)
        
        # showing
        ad = 0.2
        plt.axis([X0.min()-ad, X0.max()+ad, X1.min()-ad, X1.max()+ad])
        
        if(show):
            plt.title(titulo)
            plt.show()
            
    def run(self, labels, stream, cross_validation=False, fold=5, qtd_folds=30, plot=False):
        '''
        method to run the algorithm
        :labels: the existent labels on the stream
        :stream: the data stream
        '''

        ####### 1. Adjusting the labels of the data stream ###################################
        # salvando o stream e o tamanho do batch
        self.STREAM = al.adjustStream(labels, stream)
        ####### 1. ###########################################################################


        
        ####### 2. Training initial model (Line 1) ###########################################        
        # obtaining the initial window to train the model
        Train = self.STREAM[:self.TRAIN_SIZE]
        # training the classifier
        self.CLASSIFIER = self.buildLearner(Train)
        
        # to plot the classifier and patterns
        if(plot):
            self.plotLearnedBoundaries(self.CLASSIFIER, Train[:, 0:-1], Train[:,-1], "Step: "+str(1))
        
        # variable to store the new instances
        Train = []
        # variable to store the new instances
        Buffer = []
        ######################################################################################
    
    
    
        ####### 3. Applying the algorithm on data stream #####################################
        # for para percorrer a stream
        for i, X in enumerate(self.STREAM[self.TRAIN_SIZE:]):
            
            
            ##### 3.1. Applying cross validation ####################################
            # to use the cross validation
            run=False
            if(cross_validation and self.cross_validation(i, qtd_folds, fold)):
                run = True
            
            # to execute the prequential precedure
            if(run):
                # split the current example on pattern and label
                x, y = np.asarray([X[0:-1]]), np.asarray([int(X[-1])])
            ##### 3.1. ###############################################################
        
        
                ######### 3.2 Predict and Store new instances ##########################
                # predicting the class of the new instance (Line 3)
                y_pred = self.CLASSIFIER.predict(x)
                
                # storing the predictions
                self.PREDICTIONS.append(y_pred[0])
                self.TARGET.append(y[0])
                ######### 3.2 ##########################################################
                
                
                
                ##########3.3 getting new data #########################################
                # (Line 4) 
                if(len(Train) <= self.TRAIN_SIZE):
                    # getting the new instances and storing in buffer (Line 5)
                    Train = self.incrementWindow(Train, X)
                
                # (Line 6)
                else:
                    # getting the new instances and storing in buffer (Line 7)
                    Buffer = self.incrementWindow(Buffer, X)
                ##########3.3 ###########################################################
                
                
                
                ######### 3.4 identifying drifts (Line 8) ###############################
                if(len(Buffer) >= self.TRAIN_SIZE):
                    
                    # getting the data drifted (Line 9)
                    D1_dec, D1_sta, D1_inc, D2_dec, D2_sta, D2_inc = self.drift_detector.detect(Train, Buffer)
                    
                    # data adapted to train the new learner (Line 10 and 11)
                    Train = self.mergeData(D1_dec, D1_sta, D1_inc, D2_dec, D2_sta, D2_inc, Train, Buffer)
                    
                    # training the classifier (Line 12)
                    self.CLASSIFIER = self.buildLearner(Train)

                    # to plot the classifier and patterns
                    if(plot):
                        self.plotLearnedBoundaries(self.CLASSIFIER, Train[:, 0:-1], Train[:,-1], "Adaptation-Step: "+str(i))
                                            
                    # variable to store the new instances
                    Train = []
                    
                    # variable to store the new instances
                    Buffer = []
                ######### 3.4 ##########################################################
        
        
                ######### 3.5 doing the online update ##################################
                # (Line 13)
                else:
                    # (Line 14) 
                    self.CLASSIFIER.partial_fit(x, y)
                ######### 3.5 ##########################################################
                
                
                # print the current process
                self.printIterative(i)
        ####### 3. ############################################################################
        
def main():
    
    #1. importando o dataset
    i = 4
    #dataset = ['noaa', 'elec', 'PAKDD']
    #labels, _, stream_records = ARFFReader.read("../data_streams/real/"+dataset[i]+".arff")
    dataset = ['circles', 'sine1', 'sine2', 'virtual_5changes', 'virtual_9changes', 'SEA', 'SEARec']
    labels, _, stream_records = ARFFReader.read("../data_streams/_synthetic/"+dataset[i]+".arff")
    #stream_records = stream_records[1700:]
    
    # running the algorithm
    preq = LDD_DSDA(train_size=50)
    preq.run(labels, stream_records, cross_validation=True, fold=1, qtd_folds=30, plot=True)
    
    # printando a acuracia final do sistema
    print("Acuracia: ", preq.accuracyGeneral())
    preq.plotAccuracy()
    
    # storing only the predictions
    import pandas as pd
    df = pd.DataFrame(data={'predictions': preq.PREDICTIONS, 'target': preq.TARGET})
    df.to_csv("../projects/"+preq.NAME+"-"+dataset[i]+"_8.csv")
    
if __name__ == "__main__":
    main()        
