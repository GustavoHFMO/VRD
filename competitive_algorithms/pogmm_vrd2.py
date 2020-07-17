'''
Created on 22 de ago de 2018
@author: gusta
'''

from competitive_algorithms.prequential_super import PREQUENTIAL_SUPER
from data_streams.adjust_labels import Adjust_labels
from streams.readers.arff_reader import ARFFReader
from gaussian_models.gmm_unsupervised import GMM
from gaussian_models.gmm_unsupervised import Gaussian
from sklearn.neighbors import NearestNeighbors
from detectors.eddm import EDDM
from sklearn.metrics.classification import accuracy_score
al = Adjust_labels()
import numpy as np
import warnings
import copy
warnings.simplefilter("ignore")
np.random.seed(0)

class GMM_KDN(GMM):
    def __init__(self, noise_threshold=0.8, n_vizinhos=5, kmax=2, emit=5, stop_criterion=True):
        '''
        Constructor of GMM_VD model
        :kdn_train: to activate the use of kdn on training
        :criacao: to activate the creation of gaussians throught the stream
        :tipo_atualizacao: type of update used
        :noise_threshold: the value to define an noise
        :kmax: max number of gaussian used per class
        :n_vizinhos: number of neighboors used on kdn
        '''
        
        self.noise_threshold = noise_threshold
        self.n_vizinhos = 1+n_vizinhos
        self.Kmax = kmax
        self.emit = emit
        self.stop_criterion = stop_criterion
        
    '''
    KDN PRE-PROCESSING
    '''
        
    def easyInstances(self, x, y, limiar, n_vizinhos):
        '''
        Method to return a subset of validation only with the easy instacias
        :param: x: patterns
        :param: y: labels
        :return: x_new, y_new: 
        '''
        
        # to guarantee
        classes1, _ = np.unique(np.asarray(y), return_counts=True)
        
        # computing the difficulties for each instance
        dificuldades = self.kDN(x, y)
        
        # to guarantee that will be there at least one observation
        cont = 0
        for i in dificuldades:
            if(i < limiar):
                cont += 1
        if(cont <= n_vizinhos):
            limiar = 1
            
        # variables to save the new instances
        x_new = []
        y_new = []
         
        # saving only the easy instances
        for i in range(len(dificuldades)):
            if(dificuldades[i] < limiar):
                x_new.append(x[i])
                y_new.append(y[i])
                
        # receiving the number of classes
        classes, qtds_by_class = np.unique(np.asarray(y_new), return_counts=True)
        
        # returning only the easy instances
        # condition of existence
        if(len(qtds_by_class) != len(classes1) or True in (qtds_by_class < 5)):
            return x, y, classes1
        else:
            return np.asarray(x_new), np.asarray(y_new), classes
    
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
    
    def kDNIndividual(self, x_query, y_query, x_sel, y_sel, plot=False):
        '''
        Metodo para computar o grau de dificuldade de uma observacao baseado em um conjunto de validacao
        :param: x_query: padrao a ser consultado
        :param: x: padroes dos dados
        :param: y: respectivos rotulos
        :return: dificuldade: flutuante com a probabilidade da instancia consultada 
        '''
    
        # defining the neighboors
        nbrs = NearestNeighbors(n_neighbors=self.n_vizinhos, algorithm='ball_tree').fit(x_sel)
        
        # consulting the next neighboors
        _, indices = nbrs.kneighbors([x_query])
        # removing the query instance
        indices = indices[0][1:]
        
        # verifying the labels
        cont = 0
        for j in indices:
            if(y_sel[j] != y_query):
                cont += 1
                    
        # computing the hardness
        hardness = cont/(self.n_vizinhos-1)
        
        #====================== to plot the neighboors ===================================
        if(plot):
            self.plotInstanceNeighboors(x_query, y_query, hardness, indices, x_sel)
        #==================================================================================
            
        # returning the hardness
        return hardness
    
    
    '''
    SUPERVISED LEARN
    '''
    
    def fit(self, train_input, train_target, noise_threshold=False, n_vizinhos=False):
        '''
        method to fit the data to be clusterized
        :param: train_input: matrix, the atributes must be in vertical, and the examples in the horizontal - must be a dataset that will be clusterized
        :param: train_target: vector, with the labels for each pattern input
        :param: Kmax: number max of gaussians to test. Default 4
        :param: restarts: integer - number of restarts. Default 1
        :param: iterations: integer - number of iterations to trains the gmm model. Default 30
        '''
        
        # initialization
        if(noise_threshold==False):
            noise_threshold = self.noise_threshold
        elif(n_vizinhos==False):
            n_vizinhos = self.n_vizinhos
        
        # getting only the easy instances
        self.train_input, self.train_target, unique = self.easyInstances(train_input, train_target, noise_threshold, n_vizinhos)
        
        # updating the variables to use for plot
        self.L = len(unique)
        self.unique = unique
        
        # instantiating the gaussians the will be used
        self.gaussians = []
         
        # creating the optimal gaussians for each class
        for y_true in unique:
            
            # dividing the patterns by class
            x_train, _ = self.separatingDataByClass(y_true, self.train_input, self.train_target)

            # training a gmm
            self.trainGaussians(x_train, y_true, kmax=self.Kmax)
        
        # updating the gaussian weights          
        self.updateWeight()
                            
    def separatingDataByClass(self, y_true, x_train, y_train):
        '''
        method to separate data by class
        :y_true: label to be separeted
        :x_train: patterns
        :y_train: labels
        :return: x_train, y_train corresponding y_true
        '''
        
        # getting by class
        X_new, Y_new = [], []
        for x, y in zip(x_train, y_train):
            if(y == y_true):
                X_new.append(x)
                Y_new.append(y)
                
        # returning the new examples
        return np.asarray(X_new), np.asarray(Y_new)
    
    def trainGaussians(self, data, label, type_selection="AIC", kmax=2):
        '''
        method to train just one class
        :label: respective class that will be trained
        :data: data corresponding label
        :type_selection: AIC or BIC criterion
        '''
        
        # EM with AIC applied for each class
        gmm = self.chooseBestModel(data, type_selection, kmax, 1, self.emit, self.stop_criterion)

        # adding to the final GMM the just trained gaussians
        self.addGMM(gmm, label)
        
        # returning the gmm
        return gmm

    def updateWeight(self):
        '''
        Method to update the mix
        '''
        
        # computing the density
        sum_dens = 0 
        for g in self.gaussians:
            sum_dens += g.dens
        if(sum_dens == 0.0): sum_dens = 0.01
        
        # for each gaussian computing its weight
        for i in range(len(self.gaussians)):
            self.gaussians[i].mix = self.gaussians[i].dens/sum_dens


    '''
    SUPPORT METHODS
    '''
    
    def addGMM(self, gmm, y_true):
        '''
        Method to add a new gmm in the final GMM
        :y: respective label of GMM
        :gmm: gmm trained
        '''

        # storing the gaussians            
        for gaussian in gmm.gaussians:
            gaussian.label = y_true 
            self.gaussians.append(gaussian)
            
        # defining the number of gaussians for the problem
        self.K = len(self.gaussians)
    
    def addGaussian(self, gaussian):
        '''
        Method to insert a new gaussian into GMM
        '''
        
        #adding the new gaussian
        self.gaussians.append(gaussian)
        
        # defining the number of gaussians for the problem
        self.K = len(self.gaussians)
        
    def removeGaussian(self, g):
        '''
        Method to remove a specifc gaussian
        :g: index of the gaussian
        '''
        
        # verifying the gaussians that will be removed
        del self.gaussians[g]
        
        # defining the number of gaussians for the problem
        self.K = len(self.gaussians)
        
class OGMM(GMM_KDN):
    def __init__(self):
        super().__init__()
        self.noise_threshold = 0.5
        self.n_vizinhos = 1+9
        

    '''
    METHOD INITIALIZATION
    '''
    
    def start(self, train_input, train_target, noise_threshold=False, n_vizinhos=False):
        '''
        method to fit the data to be clusterized
        :param: train_input: matrix, the atributes must be in vertical, and the examples in the horizontal - must be a dataset that will be clusterized
        :param: train_target: vector, with the labels for each pattern input
        :param: Kmax: number max of gaussians to test. Default 4
        :param: restarts: integer - number of restarts. Default 1
        :param: iterations: integer - number of iterations to trains the gmm model. Default 30
        '''
        
        # training the GMM
        self.fit(train_input, train_target, noise_threshold, n_vizinhos)
         
        ######## activating online mechanisms ########           
        # defining the theta
        self.computeTheta()
        
        # defining the sigma
        self.computeSigma()
    
    def computeTheta(self, plot=False):
        '''
        Method to define the theta value
        '''
        
        # creating the variable theta
        self.min_classes = [0] * len(self.unique)
        
        # creating the optimal gaussians for each class
        for y_true in self.unique:
            # converting the type
            y_true = int(y_true)
            # getting data by class 
            x_train, y_train = self.separatingDataByClass(y_true, self.train_input, self.train_target)
            # getting theta by class
            self.min_classes[y_true] = self.computeThetaByClass(y_true, x_train, y_train)
        
        # defining the rule for theta
        self.theta = self.min_classes
        #self.theta = np.min(self.min_classes)

        # to plot        
        if(plot):
            self.plotGmmTheta() 
      
    def computeThetaByClass(self, y_true, x_train, y_train, plot=False):
        '''
        method to verify the furthest observation by class
        :y_true: the class that will be updated
        :x_train: the patterns of y_true
        '''
        
        # computing the pertinence
        pertinencia = []
        for x in x_train:
            pertinencia.append(self.predictionProb(x))
        
        # to plot theta
        if(plot):
            return x_train[np.argmin(pertinencia)], y_train[np.argmin(pertinencia)]
        else:
            # min by class
            return np.min(pertinencia)
      
    def computeSigma(self):
        '''
        Method to define the sigma value
        '''

        # computing the max and minimum value for the training set
        x_max = np.max(self.train_input)
        x_min = np.min(self.train_input)
         
        # computing the sigma value
        self.sigma = (x_max - x_min)/20
        
    '''
    VIRTUAL ADAPTATION
    '''
        
    def virtualAdaptation(self, x, y, W, t, plot):
        '''
        method to update an gaussian based on error
        :x: current pattern
        :y: true label of pattern
        :W: validation dataset
        :t: time
        '''
        
        if(self.noiseFiltration(x, y, W)):
            
            # find the nearest gaussian
            prob, gaussian = self.nearestGaussian(x, y)
    
            # to update the nearest gaussian            
            self.updateGaussian(x, gaussian)
            
            # to create a gaussian
            #if(self.theta > prob):
            if(self.theta[y] > prob):
                if(plot):
                    self.plotGmm("Criacao antes")
                    self.kDNIndividual(x, y, W[:,0:-1], W[:,-1], plot=True)                    
                self.createGaussian(x, y)
                self.updateTheta(y, prob)
                if(plot):
                    self.plotGmm("Criacao depois")
    
    def noiseFiltration(self, x, y, W, plot=False):
        '''
        Method to filter noisy observations
        :x: current pattern
        :y: true label of pattern
        :W: validation dataset
        '''
        
        # adjusting the window
        W = np.asarray(W)
                                
        # updating the data for train and target
        self.train_input = W[:,0:-1]
        self.train_target = W[:,-1]
            
        # to verify if the instance is an noisy
        if(self.kDNIndividual(x, y, self.train_input, self.train_target, plot) < self.noise_threshold):
            return True
        else:
            return False
     
     
    '''
    ONLINE LEARNING
    '''
        
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
                z[i] = self.conditionalProbability(x, i)

        # nearest gaussian
        gaussian = np.argmax(z)
        
        # returning the probability and the nearest gaussian
        #return z[gaussian], gaussian
        return np.sum(z), gaussian
        
    def updateGaussian(self, x, gaussian):
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
            self.gaussians[i].dens += self.posteriorProbability(x, i)
        
    def updateMean(self, x, gaussian):
        '''
        Method to update the mean of a gaussian i
        :x: pattern
        :gaussian: gaussian number
        return new mean
        '''
        
        # computing the new mean
        part1 = self.posteriorProbability(x, gaussian)/self.gaussians[gaussian].dens
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
        part5 = self.posteriorProbability(x, i)/self.gaussians[i].dens
        
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


    '''
    CREATING GAUSSIANS ONLINE
    '''
    
    def createGaussian(self, x, y):
        '''
        method to create a new gaussian
        :x: observation 
        :y: label
        '''
        
        # mu
        mu = x
        
        # covariance
        cov = (self.sigma**2) * np.identity(len(x))
        
        # label
        label = y
        
        # new gaussian
        g = Gaussian(mu, cov, 1, label)
        
        # adding the new gaussian in the system
        self.gaussians.append(g)
        
        # adding the new theta
        self.min_classes.append(0)
        
        # adding 
        self.K += 1
        
        # updating the density of all gaussians
        self.updateLikelihood(x)
        
        # updating the weights of all gaussians
        self.updateWeight()
        
    def updateTheta(self, y, prob):
        '''
        method to update the theta value
        :x: current observation
        :y: respective class
        '''
        
        # updating the prob for the current class
        self.min_classes[y] = prob
        
        # updating the theta value
        #self.theta = np.min(self.min_classes)
        self.theta = self.min_classes
       
class POGMM_VRD(PREQUENTIAL_SUPER):
    def __init__(self, batch_size=200, num_models=5, P=10, pool_exclusion="older", tax=0.2, step=1, pool_training=True, pool_reusing=True):
        '''
        method to use an gmm with a single train to classify on datastream
        :param: classifier: class with the classifier that will be used on stream
        :param: stream: stream that will be tested
        '''

        # INSTANTIATING THE BASE MEHTODS
        self.CLASSIFIER = OGMM()
        self.DETECTOR = EDDM(min_instance=batch_size)
        self.BASE_DETECTOR = EDDM(min_instance=int(batch_size/5))
        
        # PARAMETERS TO TRAIN
        self.TRAIN_SIZE = batch_size
        self.VAL_SIZE = int(self.TRAIN_SIZE * tax)
        
        # PARAMETER TO EVALUATE GAUSSIANS
        self.STEP = int(self.TRAIN_SIZE/step)
        self.FIRST_DRIFT = None
        
        # PARAMETER TO DEFINE THE FRAMEWORK
        self.NUM_MODELS = num_models
        self.P = P
        self.POOL_EXCLUSION = pool_exclusion
        
        # VARIABLES FOR RESULTS
        self.LOSS_STREAM = []
        self.DETECTIONS = [0]
        
        # AUXILIAR VARIABLES
        self.DRIFT_SIGNAL = False
        self.WARNING_SIGNAL = False
        self.NORMAL_SIGNAL = False
        self.COLLECTION_DATA = False
        
        # VARIABLES TO ANALYZE THE STRATEGY
        self.POOL_TRAINING = pool_training
        self.POOL_REUSING = pool_reusing
        
        # METHOD NAME
        #self.NAME = "POGMM_VRD-pool-"+str(self.POOL_TRAINING)+"-reuse-"+str(self.POOL_REUSING)
        self.NAME = "POGMM_VRD"
    
        # auxiliar variable
        self.PREDICTIONS = []
        self.TARGET = []
        self.count = 0
        
    '''
    METHODS TO TRAIN THE CLASSIFIERS
    '''
    
    def trainClassifier(self, W):
        '''
        method to train an classifier on the data into W
        :param: W: set with the patterns that will be trained
        '''

        # index to split the data into training and validation sets
        index = len(W) - self.VAL_SIZE
        
        # data for test
        W_train = W[:index]
        
        # split patterns and labels 
        x_train, y_train = W_train[:,0:-1], W_train[:,-1]
        
        # fitting the dataset
        self.CLASSIFIER.start(x_train, y_train)
            
        # returning the new classifier        
        return self.CLASSIFIER
    
    def getClassifierAndPool(self, num_models, W, plot):
        '''
        Method to generate the pool and online classifier
        :P: the number of classifiers that will be generated
        :W: data for training the models
        '''
        
        ################ generating the pools ######################################
        # index to split the data into training and validation sets
        index = len(W) - self.VAL_SIZE
        
        # split patterns and labels
        W_train = W[:index]
        
        # gerando o pool
        POOL_models = self.generateIndividualPool(num_models, W_train, plot)
        ############################################################################
        
        
        ################# evaluating the pools #####################################
        # data for test
        W_val = W[index:]
        
        # getting the best classifier
        best_model = self.getBestModel(POOL_models, W_val, plot)
        ############################################################################
        
        
        # returning the classifier and pool
        return best_model, POOL_models
    
    def getBestModel(self, POOL, W_val, plot):
        '''
        Method to get the best model of the pool
        '''
        
        # splitting the data into patterns and labels
        x_val, y_val = W_val[:,0:-1], W_val[:,-1]
        
        # evaluating all models 
        for i, model in enumerate(POOL):
            
            # initial model
            if(i==0):
                best_acc = accuracy_score(model.predict(x_val), y_val)
                best_index = i
                acc = best_acc
            
            # other models
            else:
                acc = accuracy_score(model.predict(x_val), y_val)
            
                # to store the best model
                if(acc >= best_acc):
                    best_acc = acc
                    best_index = i

            # to plot
            if(plot):
                # changing the dataset that will be ploted
                model.train_input = x_val
                model.train_target = y_val
                model.plotGmm("model ["+str(i)+"]", acc)
                print("[", i, "]: ", acc)
        
        if(plot):
            # best model
            print("melhor modelo [", best_index, "]: ", best_acc)
            POOL[best_index].plotGmm("melhor modelo ["+str(best_index)+"]", best_acc)
        
        # returning the best model
        return copy.deepcopy(POOL[best_index])
                
    '''
    METHODS TO MANAGE THE DATA
    '''
      
    def transferKnowledgeWindow(self, W, W_warning):    
        '''
        method to transfer the patterns of one windown to other
        :param: W: window that will be updated
        :param: W_warning: window with the patterns that will be passed
        '''
        
        W = W_warning
        
        return W
    
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
    
    def manageWindowWarning(self, W, x):
        '''
        method to reset the window
        :param: W: window that will be updated 
        '''
        
        # incrementing the window
        W = self.incrementWindow(W, x)
            
        # reseting old observations
        #if(len(W) > self.TRAIN_SIZE/2):
        #    W = self.resetWindow(W)
        
        # returning the new window
        return W
     
    def resetWindow(self, W):
        '''
        method to reset the window
        :param: W: window that will be updated 
        '''
        
        return np.array([])

    '''
    METHODS MANAGE DETECTORS BY CLASS
    '''
    
    def activateStrategiesDrift(self, warning, drift):
        '''
        Method to manage the drift and warning signal
        '''
        
        if(drift):
            drift_signal = True
            warning_signal = False
            normal_signal = False
            return drift_signal, warning_signal, normal_signal
        
        elif(warning):
            drift_signal = False
            warning_signal = True
            normal_signal = False
            return drift_signal, warning_signal, normal_signal
        
        else:
            drift_signal = False
            warning_signal = False
            normal_signal = True
            return drift_signal, warning_signal, normal_signal
                    
    def startDetectors(self, classifier, W):
        
        # split patterns and labels
        x_train, y_train = W[:,0:-1], W[:,-1]
        
        # creating one detector per gaussian
        self.DETECTORS = [copy.deepcopy(self.BASE_DETECTOR) for _ in range(len(classifier.gaussians))]
        self.STATE_DETECTORS = ["Normal"] * len(self.DETECTORS)
        
        
        # updating each detector
        for x, y in zip(x_train, y_train):
            gaussian, yi = classifier.predict_gaussian(x)
            
            # verifying the classification
            if(yi == y):
                flag = False
            else:
                flag = True
                
            # updating the detector
            self.DETECTORS[gaussian].run(flag) 

    def monitorDetectors(self, gaussian, y_true, y_pred):
        '''
        method to check the correct drift detector
        :y_true: the true label
        :y_pred: the prediction
        '''
        
        # monitoring per class
        warning, drift = self.DETECTORS[gaussian].detect(y_true, y_pred)
        
        # storing the state
        if(drift):
            self.STATE_DETECTORS[gaussian] = "Drift"
        elif(drift==False and warning):
            self.STATE_DETECTORS[gaussian] = "Warning"
            
    def checkingDetectors(self):
        '''
        Method to verify if there are degraded models
        '''
        
        return [True if detector == "Drift" else False for detector in self.STATE_DETECTORS]
    
    def checkingNewGaussians(self, classifier, pool_gaussians, train, plot):
        '''
        method check if another gaussian was created in order to create another detector
        :classifier: current classifier
        :x: the respective pattern
        :y: the respective label
        '''
        
        # creating a new detector if a new gaussian was created
        if(len(self.DETECTORS) != len(classifier.gaussians)):
            # creating the detector for the last created gaussian
            self.addDetector()
            # getting the last created gaussian 
            gaussian = self.CLASSIFIER.gaussians[-1]
            # inserting that gaussian into a pool
            return self.insertGaussian(pool_gaussians, gaussian.label, gaussian, train, plot)
        
        # case not return the same
        else:
            return pool_gaussians
        
    def addDetector(self):
        '''
        method to create another detector
        '''
        
        self.DETECTORS.append(copy.deepcopy(self.BASE_DETECTOR))
        self.STATE_DETECTORS.append("Normal")
        
    def removeDetector(self, index):
        '''
        method to remove the variables of the detector
        :index: the respective index of the detector
        '''
        
        del self.DETECTORS[index]
        del self.STATE_DETECTORS[index]
    
    def printDetectorsState(self, i, classifier):
        for j in range(len(self.STATE_DETECTORS)):
            print("[",i,"][gaussian ",j,": "+self.STATE_DETECTORS[j]+"][class ", classifier.gaussians[j].label,"][mix", classifier.gaussians[j].mix, "]")
    
    '''
    METHOD TO GENERATE AND SUBSTITUTE THE OBSOLETE GAUSSIANS
    '''
    
    def generateIndividualPool(self, num_models, W_train, plot):
        '''
        Method to generate a pool of classifiers
        '''
        
        # split patterns and labels
        x_train, y_train = W_train[:,0:-1], W_train[:,-1]
        
        # pool of models
        POOL = []
        
        # different thresholds to generate different datasets
        thresholds = [0.8] + [i/num_models for i in range(1, num_models+1)]
        
        # to guarantee that the number of neighboors is enough
        if(num_models < 5):
            num_models = 5
        
        # for to generate the models
        for i in range(len(thresholds)):
            if(plot):
                print("Training model: ", i)
            model = copy.deepcopy(self.CLASSIFIER)
            model.start(x_train, y_train, thresholds[i], num_models)
            if(plot):
                model.plotGmm(str(thresholds[i]))
            POOL.append(model)
            
        # returning the pool
        return POOL
    
    def generatePoolByClass(self, POOL_models, W, plot):
        '''
        Method to separate each gaussian by its class
        '''
        
        
        ############################# to start the pool #################################
        # creating the pool
        POOL_gaussians = []
        # for to iterate over the existing classes
        for label in POOL_models[0].unique:
            # for to start the pool of gaussians
            group = []
            for gaussian in POOL_models[0].gaussians:
                # to store the gaussians in each pool
                if(gaussian.label == label):
                    group.append(gaussian)
            # storing the pools by gaussians
            POOL_gaussians.append(group)
        #################################################################################
        
        
        ############################ adding the other models ############################
        POOL_gaussians = self.insertingPoolByClass(POOL_models[1:], POOL_gaussians, W, plot)
        #################################################################################
        
        
        # returning the pool with gaussians for each class           
        return POOL_gaussians
    
    '''
    METHODS TO  EVALUATE GAUSSIANS FROM POOL
    '''
    
    def startEvaluation(self, drifts, t):
        '''
        Method to start the time to evaluate the gaussians from pool
        :drifts: gaussians with drift
        :t: time step
        '''
        
        if((self.FIRST_DRIFT == None) and (True in drifts)):
            self.FIRST_DRIFT = t
            
    def timeEvaluation(self, drift, t):
        '''
        Method to start the time to evaluate the gaussians from pool
        :drifts: gaussians with drift
        :t: time step
        '''
        
        # activating the strategy based on steps
        if(self.FIRST_DRIFT == t):
            # updating the next update
            self.FIRST_DRIFT += self.STEP
            #print("Atualizacao local: ", t)
            return True
        else:
            return False
    
    def evaluateGaussians(self, t, drifts, classifier, W, POOL_gaussians, plot):
        '''
        Method to evaluate all gaussians in the pool to substitute the obsolete gaussian
        :classifier: the current classifier
        :W: data to evaluate the models
        :plot: to plot some graphs
        '''
        
        ###################### variables to use below ##################### 
            # split patterns and labels
        X, Y = W[:,0:-1], W[:,-1]
            
        # updating the set of the classifier
        classifier.train_input = X
        classifier.train_target = Y
        ###################################################################
    
    
        ########################## obsolete model #########################
            # storing the best accuracy for the received data
        best_model = classifier
        best_accuracy = accuracy_score(best_model.predict(X), Y)
    
        # storing the base model in another variable to be changed
        model_wo_gaussian = copy.deepcopy(best_model)

        # plotting before to remove
        if(plot):
            print("[ base ]: ", best_accuracy)
            classifier.plotGmmDrift("Antes de remover: "+str(t), drifts, best_accuracy)
        ###################################################################
            
            
        ################# removing the obsolete gaussian #############################
        # searching the obsolete gaussian
        for i in range(len(drifts)-1, -1, -1):
            if(drifts[i]):
                    
                ######### getting the label of the gaussian with drift ###########
                label = int(model_wo_gaussian.gaussians[i].label) 
                # removing the gaussian with drift
                model_wo_gaussian.removeGaussian(i)
                # plotting the model without gaussian
                ##################################################################
                    
                    
                    
                ##################### creating new models #########################
                for j, gaussian in enumerate(POOL_gaussians[label]):
                    #creating a copy of the gmm without gaussian
                    new_model = copy.deepcopy(model_wo_gaussian)
                                                
                    ############# adding a gaussian #########################
                    # adding new gaussian
                    new_model.addGaussian(gaussian)
                    new_model.updateWeight()
                        
                    #  evaluating the accuracy of the new model
                    acc = accuracy_score(new_model.predict(X), Y)
                    
                    if(plot):
                        # to follow the execution
                        print("Gaussian drift["+str(i)+"] add["+str(j)+"]: ", acc)
                        new_model.plotGmmDriftAdd("Depois de adicionar: ["+str(j)+"]", len(drifts)-1, acc)
                        
                    # comparison to store the best model
                    if(acc > best_accuracy):
                        best_model = copy.deepcopy(new_model)
                        best_accuracy = acc
                    ##########################################################
                        
                    
                ####################################################################
                        
                    
                ####################################################################
                # fixing the best model
                model_wo_gaussian = copy.deepcopy(best_model)
                    
                # plotting the best model
                if(plot):
                    # best model
                    print("Best Model: "+str(t), best_accuracy)
                    best_model.plotGmm("Best Model: "+str(t), best_accuracy)
                ####################################################################
    
        ################# after removing the obsolete gaussian #############################
            
        # returning the best model
        return best_model
    
    '''
    METHOD TO INSERT AND REMOVE GAUSSIANS
    '''
    
    def insertingPoolByClass(self, POOL_models, POOL_gaussians, W, plot):
        '''
        method to insert new gaussians in the pool of gaussians
        '''
        
        ###################### creating a for for each new model #################
        #for to iterate over all generated gmms 
        for classifier in POOL_models:
            # for to iterate over each gaussian
            for gaussian in classifier.gaussians:
                # label of the model
                label = int(gaussian.label) 
        ##########################################################################
        
                
                ################## if there are space include the model ##########
                # analyzing if the pull is totally filled
                if(len(POOL_gaussians[label]) < self.P):
                    # adding the new gaussian
                    POOL_gaussians[label].append(gaussian)
                ##################################################################
                
                
                ########### if there arent space does not include the model ######
                # excluding the oldest gaussian to add the new one
                else:
                    POOL_gaussians = self.insertGaussian(POOL_gaussians, label, gaussian, W, plot)
                ##################################################################
                            
        # returning the update pool
        return POOL_gaussians
    
    def insertGaussian(self, POOL_gaussians, label, gaussian, W, plot):
        '''
        Method to select by itself the correct way to exclude gaussian
        :POOL_gaussians: the pool of gaussians
        :y_true: the class of the current gaussian
        :gaussian: the gaussian to be inserted
        :W: the current data
        '''
        
        if(self.POOL_EXCLUSION=="older"):
            POOL_gaussians = self.removeOldestGaussian(POOL_gaussians, label, gaussian, W, plot)
        elif(self.POOL_EXCLUSION=="distance"):
            POOL_gaussians = self.removeClosestGaussian(POOL_gaussians, label, gaussian, W, plot)
        elif(self.POOL_EXCLUSION=="pertinence"):
            POOL_gaussians = self.removePoorGaussian(POOL_gaussians, label, gaussian, W, plot)
        
        return  POOL_gaussians
        
    def removeOldestGaussian(self, POOL_gaussians, label, gaussian, W, plot):
        '''
        method to exclude the oldest gaussian in the pool
        :POOL_gaussians: the pool of gaussians
        :label: the class of the current gaussian
        :gaussian: the gaussian to be inserted
        '''
        
        # split patterns and labels
        x_train, y_train = W[:,0:-1], W[:,-1]
        
        # excluding the oldest gaussian
        del POOL_gaussians[label][0]
        
        # adding the more recent gaussian
        POOL_gaussians[label].append(gaussian)


        if(plot):           
            # to plot
            model_test = OGMM()
            model_test.gaussians = []
            model_test.gaussians.append(POOL_gaussians[label][0])                         
            model_test.train_input = x_train
            model_test.train_target = y_train
            model_test.unique = np.unique(np.asarray(y_train))
            model_test.plotGmm("Pool Exclusion")
            
                            
        # returning the new pool
        return POOL_gaussians
    
    def removePoorGaussian(self, POOL_gaussians, label, gaussian, W, plot):
        '''
        method to exclude the gaussian with the worst accuracy
        :POOL_gaussians: the pool of gaussians
        :y_true: the class of the current gaussian
        :gaussian: the gaussian to be inserted
        :W: the current data
        '''
        
        # split patterns and labels
        x_train, y_train = W[:,0:-1], W[:,-1]
        
        # accuracies
        pertinences = []
        # removing the worse
        for model in POOL_gaussians[label]: 
    
            # creating a gmm with only one gaussian
            model_test = OGMM()
            model_test.gaussians = []
            model_test.gaussians.append(model)
                        
            summ = 0
            # calculating the pertinence of each gaussian for each element of the same class                    
            for x, y in zip(x_train, y_train):
                if(y == label):
                    # adding the pertinence
                    summ += model_test.predictionProb(x)
                else:
                    # removing the pertinence
                    summ -= model_test.predictionProb(x)
            
            # storing the accuracy
            pertinences.append(summ)
             
        # removing the gaussian with the worse result
        del POOL_gaussians[label][np.argmin(pertinences)]
        
        # adding the current gaussian
        POOL_gaussians[label].append(gaussian)
        
        
        if(plot):           
            # to plot
            model_test = OGMM()
            model_test.gaussians = []
            model_test.gaussians.append(POOL_gaussians[label][np.argmin(pertinences)])                         
            model_test.train_input = x_train
            model_test.train_target = y_train
            model_test.unique = np.unique(np.asarray(y_train))
            model_test.plotGmm("Pool Exclusion")
            
        # returning the pool
        return POOL_gaussians
    
    def removeClosestGaussian(self, POOL_gaussians, label, gaussian, W, plot):
        '''
        method to exclude the gaussian with the closest distance
        :POOL_gaussians: the pool of gaussians
        :y_true: the class of the current gaussian
        :gaussian: the gaussian to be inserted
        '''
        
        # split patterns and labels
        x_train, y_train = W[:,0:-1], W[:,-1]
        
        # accuracies
        distances = []
        # removing the worse
        for model in POOL_gaussians[label]:
                        
            # mu 
            mu_distance = np.linalg.norm(model.mu-gaussian.mu)
            sigma_distance = np.linalg.norm(model.sigma-gaussian.sigma)
            d = mu_distance+sigma_distance
                        
            # storing the distance
            distances.append(d)
                    
        # removing the gaussian with the worse result
        del POOL_gaussians[label][np.argmin(distances)]
        
        # adding the current gaussian
        POOL_gaussians[label].append(gaussian)
        
        
        if(plot):           
            # to plot
            model_test = OGMM()
            model_test.gaussians = []
            model_test.gaussians.append(POOL_gaussians[label][np.argmin(distances)])                         
            model_test.train_input = x_train
            model_test.train_target = y_train
            model_test.unique = np.unique(np.asarray(y_train))
            model_test.plotGmm("Pool Exclusion")
            
                
        return POOL_gaussians
    
    '''
    SIMULATE CROSS VALIDATION IN DATA STREAMS
    '''
    
    def run(self, labels, stream, cross_validation=True, fold=5, qtd_folds=30, plot=False):
        '''
        method to run the stream
        '''
        
        
        ######################### 1. FITTING THE STREAM AND AUXILIAR VARIABLES ####################
        # defining the stream
        self.STREAM = al.adjustStream(labels, stream)

        # obtaining the initial window to train the model
        Train = self.STREAM[:self.TRAIN_SIZE]
        
        # validation windown for virtual adaptation
        Val = Train
        ######################### 1. #############################################################



        ########################### 2. STARTING THE CLASSIFIER AND DETECTOR #########################        
        
        ########### 2.1. training classifier ##########################
        if(self.POOL_TRAINING):
            # training the classifier and the pool of GMM's
            self.CLASSIFIER, POOL_models = self.getClassifierAndPool(self.NUM_MODELS, Train, plot=False)
            
            if(self.POOL_REUSING):
                # getting the pool of Gaussians
                self.POOL_gaussians = self.generatePoolByClass(POOL_models, Train, plot=False)
        else:
            self.CLASSIFIER = self.trainClassifier(Train)
        # to plot
        #if(plot):
            #self.CLASSIFIER.plotGmm("Initial Model")
        ########### 2.1. ##############################################
        
        
        ########### 2.2. initializing the detectors ###################
        # starting the general detector
        self.DETECTOR.fit(self.CLASSIFIER, Train)
        # starting the detectors
        if(self.POOL_REUSING):
            self.startDetectors(self.CLASSIFIER, Train)
        ########### 2.2. ##############################################
        
        
        ### 2.3. creating the variables to store the instances ########
        # reseting the window for train
        Train = []
        TrainWarn = []
        ########### 2.3. ##############################################
        
        ############################ 2. ##############################################################
        
        
        
        #################################### 3.SIMULATING THE STREAM ################################
        # for to operate into a stream
        for i, X in enumerate(self.STREAM[self.TRAIN_SIZE:]):
            
            # to use the cross validation
            run=False
            if(cross_validation and self.cross_validation(i, qtd_folds, fold)):
                run = True
            
            # to execute the prequential precedure
            if(run):
                # split the current example on pattern and label
                x, y = X[0:-1], int(X[-1])
        ##################################### 3. ######################################################
                
                
                
                ########################################## 4. ONLINE CLASSIFICATION ###################################
                # using the classifier to predict the class of current label
                gaussian, yi = self.CLASSIFIER.predict_gaussian(x)
                
                # storing the predictions
                self.PREDICTIONS.append(yi)
                self.TARGET.append(y)
                ########################################## 4. #########################################################
                
                
                
                ########################################## 5. VIRTUAL ADAPTATION #######################################
                # sliding the current observation into W
                Val = self.slidingWindow(Val, X)
                
                # updating the gaussian if the classifier miss
                self.CLASSIFIER.virtualAdaptation(x, y, Val, i, plot)
                
                # verifying if the number of gaussians changed
                if(self.POOL_REUSING):
                    self.POOL_gaussians = self.checkingNewGaussians(self.CLASSIFIER, self.POOL_gaussians, Val, plot=False)
                ######################################### 5. ###########################################################
                


                ################################ 7. MONITORING THE DRIFT  ##############################################
                
                ################ 7.1. monitoring the drift by GMM #############
                if(self.COLLECTION_DATA==False):
                    # getting the information from detector
                    warning, drift = self.DETECTOR.detect(y, yi)
                    # activating the most appropriate strategy
                    self.DRIFT_SIGNAL, self.WARNING_SIGNAL, self.NORMAL_SIGNAL = self.activateStrategiesDrift(warning, drift)
                ################ 7.1. #########################################    
                
                
                
                ################ 7.2. monitoring the drift by gaussians #######
                if(self.POOL_REUSING):
                    # to monitor each gaussian
                    self.monitorDetectors(gaussian, y, yi)
                    # checking the error of each gaussian
                    drifts = self.checkingDetectors()
                    # first drift
                    self.startEvaluation(drifts, i)
                    # printing each gaussian error
                    if(plot):
                        self.printDetectorsState(i, self.CLASSIFIER)
                ################ 7.2. #########################################    
                    
                ################################ 7. ######################################################################



                    
                ################################## 8. NORMAL ERROR PROCEDURES ############################################
                if(self.POOL_REUSING): 
                    if((True in drifts) and (self.NORMAL_SIGNAL) and self.timeEvaluation(drift, i)):
                        # evaluating the data into the validation window
                        #print("Normal")
                        #self.CLASSIFIER, drifts = self.removeObsoleteGaussians(i, drifts, self.CLASSIFIER, Val, plot)
                        self.CLASSIFIER = self.evaluateGaussians(i, drifts, self.CLASSIFIER, Val, self.POOL_gaussians, plot=False)
                ################################## 8. ####################################################################
                
                
                
                    
                ################################## 9. WARNING ERROR PROCEDURES ########################################### 
                if(self.WARNING_SIGNAL):
                    # managing the window warning
                    TrainWarn = self.manageWindowWarning(TrainWarn, X)
                ################################## 9. ####################################################################
                        
                        
                
                ################################## 10. DRIFT ERROR PROCEDURES ############################################
                
                ############ 10.1. stoping the monitoring of drift detect ###################
                if(self.DRIFT_SIGNAL):
                    # storing the time of change
                    self.DETECTIONS.append(i)
                    # activating the collection of new data
                    self.COLLECTION_DATA = True
                    # deactivating the drift signal
                    self.DRIFT_SIGNAL = False
                    # deactivating the warning procedures
                    self.WARNING_SIGNAL = False
                    # deactivating the warning procedures
                    self.NORMAL_SIGNAL = False
                    # reseting the window
                    Train = self.transferKnowledgeWindow(Train, TrainWarn)
                ############ 10.1. ##########################################################
                    
                    
                
                ############ 10.2. collecting new data for a reset ##########################
                if(self.COLLECTION_DATA):
                    # sliding the current observation into W
                    Train = self.incrementWindow(Train, X)
                ############ 10.2. ##########################################################
                
                
                
                ############# 10.3. evaluating before to reset ##############################
                if(self.POOL_REUSING): 
                    if((True in drifts) and (len(Train) < self.TRAIN_SIZE) and (len(Train) >= 5) and self.timeEvaluation(drift, i)):
                        #print("Drift")
                        # evaluating the data into the validation window
                        #self.CLASSIFIER, drifts = self.removeObsoleteGaussians(i, drifts, self.CLASSIFIER, Train, plot)
                        self.CLASSIFIER = self.evaluateGaussians(i, drifts, self.CLASSIFIER, Train, self.POOL_gaussians, plot=False)
                ############# 10.3. #########################################################
                
                
                
                
                ############# 10.4. system reset ############################################
                if(len(Train) >= self.TRAIN_SIZE):
                    
                    
                    ############## 10.4.1. reseting the classifier ############
                    # to plot 
                    #if(plot):
                        #self.CLASSIFIER.plotGmm("Reset antes")
                    # training the pool of classifiers
                    if(self.POOL_TRAINING):
                        self.CLASSIFIER, POOL_models = self.getClassifierAndPool(self.NUM_MODELS, Train, plot=False)
                        # adding the models generated into pool
                        if(self.POOL_REUSING):
                            self.POOL_gaussians = self.insertingPoolByClass(POOL_models, self.POOL_gaussians, Train, plot=False)
                    else:
                        self.CLASSIFIER = self.trainClassifier(Train)
                    # to plot 
                    #if(plot):
                        #self.CLASSIFIER.plotGmm("Reset depois")
                    ############## 10.4.1. #####################################
                
                    
                    ############## 10.4.2. detectors reset ######################
                    # reseting the general detector
                    self.DETECTOR.reset()
                    # fitting the general detector
                    self.DETECTOR.fit(self.CLASSIFIER, Train)
                    # starting the detectors                    
                    if(self.POOL_REUSING):
                        self.startDetectors(self.CLASSIFIER, Train)
                        self.FIRST_DRIFT = None
                    ############## 10.4.2. ######################################
                    
                    
                    ############## 10.4.3. auxiliar variables reset #############
                    # reseting the window for Train
                    Train = []
                    # stop the collection of data
                    self.COLLECTION_DATA = False
                    ############## 10.4.3. ######################################
                
                ################################## 10. ###################################################################
                
                
                
                ################################# 11. INTERATIVE PRINT #############################################################
                #self.CLASSIFIER.train_input = Val[:,0:-1]
                #self.CLASSIFIER.train_target = Val[:,-1]
                #self.CLASSIFIER.plotGmm(i)
                self.printIterative(i)
                
def main():
    
    i = 4
    #dataset = ['PAKDD', 'elec', 'noaa']
    #labels, _, stream_records = ARFFReader.read("../data_streams/real/"+dataset[i]+".arff")
    dataset = ['circles', 'sine1', 'sine2', 'virtual_5changes', 'virtual_9changes', 'SEA', 'SEARec']
    labels, _, stream_records = ARFFReader.read("../data_streams/_synthetic/"+dataset[i]+".arff")
    stream_records = stream_records[700:2600]
    
    #4. instantiate the prequetial
    preq = POGMM_VRD(batch_size=200, 
                     pool_training=True, 
                     pool_reusing=True)
    
    #5. execute the prequential
    preq.run(labels, 
             stream_records, 
             cross_validation=True, 
             qtd_folds=30, 
             fold=0, 
             plot=True)
    
    #preq.plotAccuracy()
    
    # printando a acuracia final do sistema
    print("Acuracia: ", preq.accuracyGeneral())
    
    # storing only the predictions
    import pandas as pd
    df = pd.DataFrame(data={'predictions': preq.PREDICTIONS, 'target': preq.TARGET})
    df.to_csv("../projects/"+preq.NAME+"-"+dataset[i]+".csv")
    
if __name__ == "__main__":
    main()        
           
    