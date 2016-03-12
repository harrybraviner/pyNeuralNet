#! /usr/bin/python2.7

import numpy as np
from math import exp

def sigmoid(x):
    return 1.0/(1.0 + exp(-x))

def sigmoidGradient(x):
    sig = sigmoid(x)
    return sig*(1.0 - sig)

sigmoid_broadcast = np.frompyfunc(sigmoid, 1, 1)
sigmoidGradient_broadcast = np.frompyfunc(sigmoid, 1, 1)

class pyNeuralNet:
    
    def __init__(self, inputSize, hiddenLayerSizes, outputNodes):
        if (type(inputSize) != int or inputSize <= 0) :
            raise TypeError("inputSize must be a positive int")
        if (type(hiddenLayerSizes) != list or len(hiddenLayerSizes) == 0 or any(map(lambda x : type(x) != int or x <= 0, hiddenLayerSizes))) :
            raise TypeError("hiddenLayerSizes must be a non-empty list of positive ints")
        if (type(outputNodes) != int or outputNodes <= 0) :
            raise TypeError("outputNodes must be a positive int")

        self.inputSize = inputSize
        self.numberOfLayers = len(hiddenLayerSizes) + 2 # Include both the input and output layers
        self.layerSizes = [inputSize] + hiddenLayerSizes + [outputNodes]
        self.Theta = [ np.matrix([[0.0 for j in range(columnsMinusOne+1)] for i in range(rows)], dtype = np.float64) # +1 is for the bias units
                                for columnsMinusOne, rows in zip(self.layerSizes[:-1], self.layerSizes[1:])]    # Initialise the Theta arrays to zero

    def __propForwardOneLayer__(self, inputLayer, inputVector):
        if (type(inputLayer) != int or inputLayer < 1 or inputLayer > (self.numberOfLayers-1)):
            raise TypeError("inputLayer must be an integer between 1 and {0} (the number of layers minus one) inclusive".format(self.numberOfLayers-1))
        # Just try creating an np.matrix from it and catch anything that it throws!
        try:
            inputVector = np.matrix(inputVector, dtype=np.float64)
        except Exception as e:
            raise TypeError("Failed to put inputVector into a numpy matrix of type float64.\nGot message from inner exception {0} of type {1}".format(e.message, str(type(e))))
        if (inputVector.shape == (1, self.layerSizes[inputLayer-1])) :
            inputVector = inputVector.T
        if (inputVector.shape != (self.layerSizes[inputLayer-1], 1)) :
            raise TypeError("inputVector must be contain a single column and {0} rows (the number of units in layer {1}), or be the transpose of such a vector.".format(self.layerSizes[inputLayer-1], inputLayer))

        inputVector = np.vstack(([1.0], inputVector))  # Augment with the bias unit

        outputVector = np.matrix([sigmoid(z) for z in (self.Theta[inputLayer-1] * inputVector).flat]).T
        return outputVector

    def __propForwardAToZ__(self, inputLayer, inputA):
        # Let l be inputLayer
        # This method takes a_l and return z_l+1
        if (type(inputLayer) != int or inputLayer < 1 or inputLayer > (self.numberOfLayers-1)):
            raise TypeError("inputLayer must be an integer between 1 and {0} (the number of layers) inclusive".format(self.numberOfLayers))
        try:
            inputA = np.matrix(inputA, dtype=np.float64)
        except Exception as e:
            raise TypeError("Failed to put inputA into a numpy matrix of type float64.\nGot message from inner exception {0} of type {1}".format(e.message, str(type(e))))
        if (inputA.shape == (1, self.layerSizes[inputLayer-1])):
            inputA = inputA.T
        if (inputA.shape != (self.layerSizes[inputLayer-1], 1)):
            raise TypeError("inputA must be contain a single column and {0} rows (the number of units in layer {1}), or be the transpose of such a vector.".format(self.layerSizes[inputLayer-1], inputLayer))
        
        return self.Theta[inputLayer-1] * np.vstack([[1.0], inputA])

    def __propForwardZToA__(self, inputLayer, inputZ):
        # Let l be inputLayer
        # This method takes z_l and return a_l
        if (type(inputLayer) != int or inputLayer < 2 or inputLayer > self.numberOfLayers):
            raise TypeError("inputLayer must be an integer between 1 and {0} (the number of layers) inclusive".format(self.numberOfLayers))
        try:
            inputZ = np.matrix(inputZ, dtype=np.float64)
        except Exception as e:
            raise TypeError("Failed to put inputZ into a numpy matrix of type float64.\nGot message from inner exception {0} of type {1}".format(e.message, str(type(e))))
        if (inputZ.shape == (1, self.layerSizes[inputLayer-1])):
            inputZ = inputZ.T
        if (inputZ.shape != (self.layerSizes[inputLayer-1], 1)):
            raise TypeError("inputZ must be contain a single column and {0} rows (the number of units in layer {1}), or be the transpose of such a vector.".format(self.layerSizes[inputLayer-1], inputLayer))
        
        return sigmoid_broadcast(inputZ)

    def __propForwardZToZ__(self, inputLayer, inputZ):
        # For back-propogation we need to know z, rather than a, at each layer
        # If inputlayer = l, then this takes z^l and returns z^(l+1)
        if (type(inputLayer) != int or inputLayer < 2 or inputLayer > (self.numberOfLayers-1)):
            raise TypeError("inputLayer must be an integer between 2 and {0} (the number of layers minus one) inclusive".format(self.numberOfLayers-1))
        try:
            inputZ = np.matrix(inputZ, dtype=np.float64)
        except Exception as e:
            raise TypeError("Failed to put inputZ into a numpy matrix of type float64.\nGot message from inner exception {0} of type {1}".format(e.message, str(type(e))))
        if (inputZ.shape == (1, self.layerSizes[inputLayer-1])):
            inputZ = inputZ.T
        if (inputZ.shape != (self.layerSizes[inputLayer-1], 1)):
            raise TypeError("inputZ must be contain a single column and {0} rows (the number of units in layer {1}), or be the transpose of such a vector.".format(self.layerSizes[inputLayer-1], inputLayer))

        a_l = np.vstack([[1.0], sigmoid_broadcast(inputZ)]) # Augment with a bias unit
        z_lPlusOne = self.Theta[inputLayer-1] * a_l
        
        return z_lPlusOne
        

    def __propForwardWholeNet__(self, inputVector):
        for layerIndex in range(1, self.numberOfLayers):
            inputVector = self.__propForwardOneLayer__(layerIndex, inputVector)
        return inputVector

    def __propBackOneLayer__(self, inputLayer, inputDelta, inputZ):
        # To compute delat^(l), inputLayer should be (l+1), inputDelta should be delta^(l+1), and inputZ should be z^(l)
        if(type(inputLayer) != int or inputLayer < 2 or inputLayer > self.numberOfLayers - 1):
            raise TypeError("inputLayer must be an integer between 2 and {0} (the number of layers minus 1) inclusive".format(self.numberOfLayers - 1))
        try:
            inputDelta = np.matrix(inputDelta, dtype=np.float64)
        except Exception as e:
            raise TypeError("Failed to put inputDelta into a numpy matrix of type float64.\nGot message from inner exception {0} of type {1}".format(e.message, str(type(e))))
        if (inputDelta.shape == (1, self.layerSizes[inputLayer])):
            inputDelta = inputDelta.T
        if (inputDelta.shape != (self.layerSizes[inputLayer], 1)):
            raise TypeError("inputDelta for should be delta^{0}, and should therefore have {1} elements".format(inputLayer+1, self.layerSizes[inputLayer]))
        try:
            inputZ = np.matrix(inputZ, dtype=np.float64)
        except Exception as e:
            raise TypeError("Failed to put inputZ into a numpy matrix of type float64.\nGot message from inner exception {0} of type {1}".format(e.message, str(type(e))))
        if (inputZ.shape == (1, self.layerSizes[inputLayer-1]+1)):
            inputZ = inputZ.T
        if (inputZ.shape != (self.layerSizes[inputLayer-1]+1, 1)):
            raise TypeError("inputZ for should be z^{0}, and should therefore have {1} elements".format(inputLayer, self.layerSizes[inputLayer-1]+1))

        gprime = np.matrix([sigmoidGradient(z) for z in inputZ.flat]).T

        deltaOut = np.multiply((self.Theta[inputLayer-1].T * inputDelta), gprime)   # Elementwise multiplication

        return deltaOut

    def __computeGradients__(self, trainingPredictors, trainingTargets):
        # trainingPredictors is an np.matrix where each row is a training example (transpose the row to get a1)
        # trainingTargets is an np.matrix([k1, k2, ..., kN]).T where ki is in 1, ..., outputNodes
        if (trainingTargets.shape[1] != 1):
            raise TypeError("trainingTargets should be a column vectors of integers between 1 and {0} inclusive".format(self.layerSizes[-1]))
        if (trainingPredictors.shape[0] != trainingTargets.shape[0]):
            raise TypeError("trainingPredictors and trainingTargets should have the same number of rows.")
        
        # Initialise gradients to zero
        ThetaGradient = [np.matrix([[0.0 for j in range(columnsMinusOne+1)] for i in range(rows)], dtype = np.float64) # +1 is for the bias units
                                for rows, columnsMinusOne in zip(self.layerSizes[1:], self.layerSizes[:-1])]    # Initialise the Theta arrays to zero

        N = trainingPredictors.shape[0]
        for trainingIndex in range(N):  
            a1 = trainingPredictors[trainingIndex].T
            #z2 = self.Theta[0] * np.vstack([[1.0], a1])
            aList = [a1]
            zList = []
            for l in range(2, self.numberOfLayers + 1):
                zList = zList + [self.__propForwardAToZ__(l-1, aList[-1])]
                aList = aList + [self.__propForwardZToA__(l, zList[-1])]
                #zList = zList + [self.__propForwardOneLayer__(l, zList[-1])]
            # aList = [a1, a2, ..., al]
            # zList = [z2, z3, ..., zl]
            #al = sigmoid_broadcast(zList[-1])   # Output of the net for this example
            y = np.matrix([(1.0 if (k+1 == trainingTargets[trainingIndex, 0]) else 0.0) for k in range(self.layerSizes[-1])]).T
            deltal = (aList[-1] - y)
            deltaList = [deltal]
            for l in range(self.numberOfLayers-1, 1, -1):
                deltaList = [np.multiply((self.Theta[l-1]*deltaList[-1])[1:], sigmoidGradient_broadcast(zList[l-2]))] + deltaList
            # deltaList = [delta2, delta3, ... , deltal]
            for l in range(1, self.numberOfLayers):
                ThetaGradient[l-1] = ThetaGradient[l-1] + (deltaList[l-1] * np.vstack([[1.0], aList[l-1]]).T)

        # Normalise by the number of training examples
        ThetaGradient = [grad / N for grad in ThetaGradient]

        return np.vstack([grad.flatten().T for grad in ThetaGradient])

    def __cost__(self, trainingPredictors, trainingTargets):
        # trainingPredictors is an np.matrix where each row is a training example (transpose the row to get a1)
        # trainingTargets is an np.matrix([k1, k2, ..., kN]).T where ki is in 1, ..., outputNodes
        if (trainingTargets.shape[1] != 1):
            raise TypeError("trainingTargets should be a column vectors of integers between 1 and {0} inclusive".format(self.layerSizes[-1]))
        if (trainingPredictors.shape[0] != trainingTargets.shape[0]):
            raise TypeError("trainingPredictors and trainingTargets should have the same number of rows.")

        N = trainingPredictors.shape[0]
        cost = 0.0
        for i in range(N):
            h = self.__propForwardWholeNet__(np.matrix(trainingPredictors[i,:]).T) # Output of the net on this training example
            y = np.matrix([1.0 if ((k+1)==trainingTargets[i]) else 0.0 for k in range(self.layerSizes[-1])]).T
            cost = cost - (y.T * np.log(h) + (1.0 - y).T * np.log(1.0 - h))
        cost = cost / N # Normalise
        return cost
        


    def predictClass(self, inputVector):
        outputVector = self.__propForwardWholeNet__(inputVector)

