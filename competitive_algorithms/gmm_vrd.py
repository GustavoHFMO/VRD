'''
Created on 22 de ago de 2018
@author: gusta
'''

from competitive_algorithms.prequential_super import PREQUENTIAL_SUPER
from data_streams.adjust_labels import Adjust_labels
from streams.readers.arff_reader import ARFFReader
from gaussian_models.gmm_super_old import GMM
from gaussian_models.gmm_super_old import Gaussian
from sklearn.neighbors import NearestNeighbors
from detectors.ewma import EWMA
al = Adjust_labels()
import numpy as np

np.random.seed(0)
lista = [1700, 1800, 1900, 2000]

class GMM_VD(GMM):
    def __init__(self, kdn_train=True, remocao=False, criacao=True, noise_threshold=0.8, kmax=2):
        self.kdn_train = kdn_train 
        self.remocao = remocao
        self.criacao = criacao
        self.noise_threshold = noise_threshold
        self.Kmax = kmax
        self.n_vizinhos = 5
        
        # auxiliar variable
        self.PREDICTIONS = []
        self.TARGET = []
        self.count = 0
    
    def kDNIndividual(self, x_query, y_query, x_sel, y_sel, plot=False):
        '''
        Metodo para computar o grau de dificuldade de uma observacao baseado em um conjunto de validacao
        :param: x_query: padrao a ser consultado
        :param: x: padroes dos dados
        :param: y: respectivos rotulos
        :return: dificuldade: flutuante com a probabilidade da instancia consultada 
        '''
    
        # defining the neighboors
        nbrs = NearestNeighbors(n_neighbors=self.n_vizinhos+1, algorithm='ball_tree').fit(x_sel)
        
        # consulting the next neighboors
        _, indices = nbrs.kneighbors([x_query])
            
        # verifying the labels
        cont = 0
        for j in indices[0]:
            if(all(x_query != x_sel[j]) and y_sel[j] != y_query):
                cont += 1
                    
        # computing the hardness
        hardness = cont/(self.n_vizinhos)
            
        # returning the hardness
        return hardness
    
    def kDN(self, X, Y):
        '''
        Method to compute the hardess of an observation based on a training set
        :param: X: patterns
        :param: Y: labels
        :return: dificuldades: vector with hardness for each instance 
        '''
        
        # to store the hardness
        hardness = [0] * len(Y)

        # for to compute the hardness for each instance
        for i, (x, y) in enumerate(zip(X,Y)):
            hardness[i] = self.kDNIndividual(x, y, X, Y)
        
        # returning the hardness
        return hardness
    
    def easyInstances(self, x, y, limiar):
        '''
        Metodo para retornar um subconjunto de validacao apenas com as instacias faceis
        :param: x: padroes dos dados
        :param: y: respectivos rotulos
        :return: x_new, y_new: 
        '''
        
        # computando as dificulades para cada instancia
        dificuldades = self.kDN(x, y)
        
        # to guarantee that will be there at least one observation
        cont = 0
        for i in dificuldades:
            if(i > limiar):
                cont += 1
        if(cont <= (len(dificuldades)/3)):
            limiar = 1
        
        # variaveis para salvar as novas instancias
        x_new = []
        y_new = []
         
        # salvando apenas as instancias faceis
        for i in range(len(dificuldades)):
            if(dificuldades[i] < limiar):
                x_new.append(x[i])
                y_new.append(y[i])
                
        return np.asarray(x_new), np.asarray(y_new)
    
    def fit(self, train_input, train_target, type_selection='AIC', Kmax=4, restarts=1, iterations=5):
        '''
        method to fit the data to be clusterized
        :param: train_input: matrix, the atributes must be in vertical, and the examples in the horizontal - must be a dataset that will be clusterized
        :param: train_target: vector, with the labels for each pattern input
        :param: type_selection: name of prototype selection metric. Default 'AIC'
        :param: Kmax: number max of gaussians to test. Default 4
        :param: restarts: integer - number of restarts. Default 1
        :param: iterations: integer - number of iterations to trains the gmm model. Default 30
        '''
        
        # to experiment
        if(self.Kmax != None):
            Kmax = self.Kmax
        
        # storing the patterns
        if(self.kdn_train):
            self.train_input, self.train_target = self.easyInstances(train_input, train_target, self.noise_threshold)
            
            # receiving the number of classes
            _, ammount = np.unique(self.train_target, return_counts=True)
            
            # condition of existence
            if(0 in ammount):
                self.train_input, self.train_target = train_input, train_target
            
        else:
            self.train_input, self.train_target = train_input, train_target

        # receiving the number of classes
        unique, ammount = np.unique(self.train_target, return_counts=True)
        self.L = len(unique)
        self.unique = unique
            
        # dividing the patterns per class
        classes = []
        for i in unique:
            aux = []
            for j in range(len(self.train_target)):
                if(self.train_target[j] == i):
                    aux.append(self.train_input[j])
            classes.append(np.asarray(aux))
        classes = np.asarray(classes)
        
        # variable to store the weight of each gaussian
        self.dens = []
        
        # instantiating the gaussians the will be used
        self.gaussians = []
         
        # creating the optimal gaussians for each class
        for i in range(len(classes)):

            if(ammount[i] != 0):
                # EM with BIC applied for each class
                gmm = self.chooseBestModel(classes[i], type_selection, Kmax, restarts, iterations)
            
                # storing the gaussians            
                for gaussian in gmm.gaussians:
                    gaussian.label = i 
                    self.gaussians.append(gaussian)
                            
                # storing the density of each gaussian
                for k in gmm.dens:
                    self.dens.append(k)
            else:
                continue
                        
        # defining the number of gaussians for the problem
        self.K = len(self.gaussians)
        
        # defining the weight of each gaussian based on dataset
        self.mix = [self.dens[i]/len(self.train_target) for i in range(self.K)]
     
        # intializing the matrix of weights
        self.matrixWeights = self.Estep()
        
    def predict_one(self, x):
        '''
        method to predict the class for a only pattern x
        :param: x: pattern
        :return: the respective label for x
        '''
        
        # receiving the gaussian with more probability for the pattern
        y = [0]*self.K
        for i in range(self.K):
            y[i] = self.posteriorProbability(x, i)
        gaussian = np.argmax(y)    
                    
        # returning the label
        return self.gaussians[gaussian].label
    
    def predict_one_gaussian(self, x):
        '''
        method to predict the class for a only pattern x and to show the gaussian used
        :param: x: pattern
        :return: the respective label for x
        '''
        
        # receiving the gaussian with more probability for the pattern
        y = [0]*self.K
        for i in range(self.K):
            y[i] = self.posteriorProbability(x, i)
        gaussian = np.argmax(y)    
                    
        # returning the label
        return self.gaussians[gaussian].label, gaussian
    
    def virtualAdaptation(self, x, y_true, y_pred, W, i):
        '''
        method to update an gaussian based on error 
        '''
        
        if(y_true != y_pred):
            # adjusting the window
            W = np.asarray(W)
                            
            # updating the data for train and target
            self.train_input = W[:,0:-1]
            self.train_target = W[:,-1]
            
            # find the nearest gaussian
            flag, gaussian = self.nearestGaussian(x, y_true)
                    
            # condition
            if(flag):
                # update the nearest gaussian
                self.updateGaussianIncremental(x, gaussian)
                        
            else:
                # create a gaussian
                self.createGaussian(x, y_true)
            
        if(self.remocao):
            self.removeGaussians()
        
    def nearestGaussian(self, x, y):
        '''
        method to find the nearest gaussian of the observation x
        :x: observation 
        :y: label
        '''
        
        # receiving the gaussian with more probability for the pattern
        z = [0] * len(self.gaussians)
        for i in range(len(self.gaussians)):
            if(self.gaussians[i].label == y):
                z[i] = self.gaussians[i].pdf_vector(x)

        # nearest gaussian
        gaussian = np.argmax(z)
        
        # condition to create or update gaussian
        if(z[gaussian] > 0):
            return True, gaussian
        else:
            return False, gaussian
        
    def removeGaussians(self):
        '''
        method to remove obsolete gaussians
        '''
        
        # ammount of gaussians per class
        class_gaussians = [self.gaussians[i].label for i in range(len(self.gaussians))]
        labels, ammount = np.unique(class_gaussians, return_counts=True)
            
        # to search obsolete gaussians
        erase = []
        for i in range(len(labels)):
            for j in range(len(self.gaussians)):
                if(ammount[i] > 1 and self.gaussians[j].label == labels[i] and self.mix[j] == 0.001):
                    erase.append(j)
        
        # to remove obsolete gaussians
        for i in sorted(erase, reverse=True):
            self.deleteGaussian(i)
    
    def createGaussian(self, x, y):
        '''
        method to create a new gaussian
        :x: observation 
        :y: label
        '''
        
        # mu
        mu = x
        
        # covariance
        cov = (0.5**2) * np.identity(len(x))
        
        # label
        label = y
        
        # new gaussian
        g = Gaussian(mu=mu, sigma=cov, mix=0, label=label)
        
        # adding the new gaussian in the system
        self.gaussians.append(g)
        
        # adding the new mix
        self.mix.append(1)
        
        # adding the dens
        self.dens.append(1)
        
        # adding 
        self.K += 1
        
        # updating the density of all gaussians
        self.updateLikelihood(x)
        
        # updating the weights of all gaussians
        self.updateWeight()
        
    def deleteGaussian(self, i):
        '''
        Method to delete a gaussian
        '''
        
        # deleting an gaussian
        del self.gaussians[i]
        del self.mix[i]
        del self.dens[i]
        self.K -= 1
        
    def updateGaussianIncremental(self, x, gaussian):
        '''
        method to update the nearest gaussian of x
        :x: the observation that will be used to update a gaussian
        :gaussian: the number of gaussian that will be updated  
        '''

        # updating the likelihood of all gaussians for x
        self.updateLikelihood(x)
        
        # updating the gaussian weights
        self.updateWeight()
                
        # storing the old mean
        old_mean = self.gaussians[gaussian].mu
        
        # updating the mean
        self.gaussians[gaussian].mu = self.updateMean(x, gaussian)

        # updating the covariance        
        self.gaussians[gaussian].sigma = self.updateCovariance(x, gaussian, old_mean)
        
    def updateLikelihood(self, x):
        '''
        method to update the parameter cver
        :param: x: new observation
        '''
        
        # updating the loglikelihood
        for i in range(len(self.gaussians)):
            self.dens[i] = self.dens[i] + self.posteriorProbability(x, i)
        
    def updateWeight(self):
        '''
        Method to update the mix
        '''
        
        sum_dens = np.sum(self.dens)
        if(sum_dens == 0.0): sum_dens = 0.01
        
        for i in range(len(self.gaussians)):
            self.mix[i] = self.dens[i]/sum_dens
    
    def updateMean(self, x, gaussian):
        '''
        Method to update the mean of a gaussian i
        :x: pattern
        :gaussian: gaussian number
        return new mean
        '''
        
        # computing the new mean
        part1 = self.posteriorProbability(x, gaussian)/self.dens[gaussian]
        part2 = np.subtract(x, self.gaussians[gaussian].mu)
        new = self.gaussians[gaussian].mu + (np.dot(part1, part2))
        
        # returning mean
        return new
    
    def updateCovariance(self, x, i, old_mean):
        '''
        Method to update the mean of a gaussian i
        :x: pattern
        :gaussian: gaussian number
        return new mean
        '''
        
        # equation to compute the covariance
        ######## primeira parte ##############
        # sigma passado
        part0 = self.gaussians[i].sigma
        
        # primeiro termo
        part1 = np.subtract(self.gaussians[i].mu, old_mean)
        
        # segundo termo transposto
        part2 = np.transpose([part1])
        
        # multiplicacao dos termos
        part3 = np.dot(part2, [part1])
        
        # subtracao do termo pelo antigo
        part4 = np.subtract(part0, part3)
        ########################################
        
        
        ######## segunda parte ##############
        #ajuste de pertinencia
        part5 = self.posteriorProbability(x, i)/self.dens[i]
        
        # primeiro termo
        part6 = np.subtract(x, self.gaussians[i].mu)
        
        # segundo termo transposto
        part7 = np.transpose([part6])
        
        # multiplicacao do primeiro pelo segundo
        part8 = np.dot(part7, [part6])
        
        # subtracao do sigma antigo pelos termos
        part9 = np.subtract(part8, part0)
        
        
        # multiplicacao da pertinencia pelo colchetes
        part10 = np.dot(part5, part9)
        ########################################
        
        
        #final
        covariance = np.add(part4, part10) 
        
        # returning covariance
        return covariance
    
class GMM_VRD(PREQUENTIAL_SUPER):
    def __init__(self, window_size=200, Kmax=2, virtual=True, real=True):
        '''
        method to use an gmm with a single train to classify on datastream
        :param: classifier: class with the classifier that will be used on stream
        :param: stream: stream that will be tested
        '''

        # main variables
        self.NAME = 'GMM-VRD'
        self.CLASSIFIER = GMM_VD(kdn_train=True, remocao=False, criacao=True, noise_threshold=0.7, kmax=Kmax)
        self.DETECTOR = EWMA(min_instance=window_size, c=1, w=0.5)
        self.VIRTUAL = virtual
        self.REAL = real
        self.WINDOW_SIZE = window_size
        self.LOSS_STREAM = []
        self.DETECTIONS = [0]
        self.WARNINGS = [0]
        self.CLASSIFIER_READY = True
    
    def transferKnowledgeWindow(self, W, W_warning):    
        '''
        method to transfer the patterns of one windown to other
        :param: W: window that will be updated
        :param: W_warning: window with the patterns that will be passed
        '''
        
        W = W_warning
        
        return W
    
    def manageWindowWarning(self, W, x):
        '''
        method to reset the window
        :param: W: window that will be updated 
        '''
        
        if(self.CLASSIFIER_READY):
            W = self.incrementWindow(W, x)
            
            if(len(W) > self.WINDOW_SIZE/2):
                W = self.resetWindow(W)
        
        return W
     
    def resetWindow(self, W):
        '''
        method to reset the window
        :param: W: window that will be updated 
        '''
        
        return np.array([])
    
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
    
    def slidingWindow(self, W, x):
        '''
        method to slide the window under the example
        :param: W: window that will be updated 
        :param: x: example that will be inserted 
        '''
        
        aux = [None] * len(W)
        aux[0:-1] = W[1:]
        aux[-1] = x
    
        return np.asarray(aux)

    def trainClassifier(self, W):
        '''
        method to train an classifier on the data into W
        :param: W: set with the patterns that will be trained
        '''
        
        # split patterns and labels
        x_train, y_train = W[:,0:-1], W[:,-1]
        
        # fitting the dataset
        self.CLASSIFIER.fit(x_train, y_train)

        # returning the new classifier        
        return self.CLASSIFIER
    
    def run(self, labels, stream, cross_validation=False, fold=5, qtd_folds=30):
        '''
        method to run the stream
        '''
        
        # defining the stream
        self.STREAM = al.adjustStream(labels, stream)
        
        # obtaining the initial window
        W = self.STREAM[:self.WINDOW_SIZE]
        
        # instantiating a window for warning levels 
        W_warning = []
        
        # instantiating the validation window
        if(self.VIRTUAL):
            W_validation = W 
        
        # training the classifier
        self.CLASSIFIER = self.trainClassifier(W)
        
        # fitting the detector
        self.DETECTOR.fit(self.CLASSIFIER, W) 
        
        # for to operate into a stream
        for i, X in enumerate(self.STREAM[self.WINDOW_SIZE:]):
            
            # to use the cross validation
            run=False
            if(cross_validation and self.cross_validation(i, qtd_folds, fold)):
                run = True
            
            # to execute the prequential precedure
            if(run):
                
                # debug
                if(i+self.WINDOW_SIZE in lista):
                    #self.CLASSIFIER.plotGmm(i+self.WINDOW_SIZE, self.accuracyGeneral())
                    print()
                    
                # split the current example on pattern and label
                x, y = X[0:-1], int(X[-1])
                
                if(i==3396):
                    print()
                    
                # using the classifier to predict the class of current label
                yi = self.CLASSIFIER.predict(x)
                    
                # storing the predictions
                self.PREDICTIONS.append(yi)
                self.TARGET.append(y)
                
                # activating the VIRTUAL
                if(self.VIRTUAL):
                    # sliding the current observation into W
                    W_validation = self.slidingWindow(W_validation, X)
                    
                    # updating the gaussian if the classifier miss
                    self.CLASSIFIER.virtualAdaptation(x, y, yi, W_validation, i)
    
                # verifying the claassifier
                if(self.CLASSIFIER_READY):
    
                    # sliding the current observation into W
                    W = self.slidingWindow(W, X)
                
                    # monitoring the datastream
                    warning_level, change_level = self.DETECTOR.detect(y, yi)
                            
                    # trigger the warning VIRTUAL
                    if(warning_level):
    
                        # managing the window warning
                        W_warning = self.manageWindowWarning(W_warning, X)
                        
                        # storing the time when warning was triggered
                        self.WARNINGS.append(i)
                        
                    # trigger the change VIRTUAL    
                    if(change_level):
                        
                        # storing the time of change
                        self.DETECTIONS.append(i)
                        
                        # reseting the detector
                        self.DETECTOR.reset()
                            
                        # reseting the window
                        W = self.transferKnowledgeWindow(W, W_warning)
                        
                        # reseting the classifier 
                        self.CLASSIFIER_READY = False
                
                elif(self.WINDOW_SIZE > len(W)):
                    
                    # sliding the current observation into W
                    W = self.incrementWindow(W, X)
                    
                else:
                    
                    # to remodel the knowledge of the classifier
                    self.CLASSIFIER = self.trainClassifier(W)
                        
                    #if(any([True if i+self.WINDOW_SIZE > j else False for j in lista])):
                    #    self.CLASSIFIER.plotGmm(i+self.WINDOW_SIZE, self.accuracyGeneral())
                        
                    # fitting the detector
                    self.DETECTOR.fit(self.CLASSIFIER, W) 
                            
                    # releasing the new classifier
                    self.CLASSIFIER_READY = True
                    
                # print the current process
                self.printIterative(i)
    
def main():
    
    i = 2
    #dataset = ['PAKDD', 'elec', 'noaa']
    #labels, _, stream_records = ARFFReader.read("../data_streams/real/"+dataset[i]+".arff")
    dataset = ['circles', 'sine1', 'sine2', 'virtual_5changes', 'virtual_9changes', 'SEA', 'SEARec']
    labels, _, stream_records = ARFFReader.read("../data_streams/_synthetic/"+dataset[i]+".arff")
    
    #4. instantiate the prequetial
    preq = GMM_VRD(window_size=50, Kmax=2)
    
    #5. execute the prequential
    preq.run(labels, stream_records, cross_validation=True, qtd_folds=30, fold=1)
    preq.plotAccuracy()
    
    # printando a acuracia final do sistema
    print("Acuracia: ", preq.accuracyGeneral())
    
    # storing only the predictions
    import pandas as pd
    df = pd.DataFrame(data={'predictions': preq.PREDICTIONS, 'target':preq.TARGET})
    df.to_csv("../projects/"+preq.NAME+"-"+dataset[i]+".csv")
    
if __name__ == "__main__":
    main()        
           
        
    