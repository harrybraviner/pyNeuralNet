#! /usr/bin/python2.7

import numpy as np
from math import exp

def sigmoid(x):
    return 1.0/(1.0 + exp(-x))

def sigmoidGradient(x):
    sig = sigmoid(x)
    return sig*(1.0 - sig)

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
            raise TypeError("inputLayer must be an integer between 1 and {0} (the number of layers minus one) inclusive".format(numberOfLayers-1))
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

    def __propForwardWholeNet__(self, inputVector):
        for layerIndex in range(1, self.numberOfLayers):
            inputVector = self.__propForwardOneLayer__(layerIndex, inputVector)

    def __propBackOneLayer__(self, inputLayer, inputDelta, inputZ):
        # To compute delat^(l), inputLayer should be (l+1), inputDelta should be delta^(l+1), and inputZ should be z^(l)
        if(type(inputLayer) != int or inputLayer < 2 or inputLayer > numberOfLayers - 1):
            raise TypeError("inputLayer must be an integer between 2 and {0} (the number of layers minus 1) inclusive".format(numberOfLayers - 1))
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

    def __computeGradients__(self, trainingData):
        

        # Initialise gradients to zero
        ThetaGradient = [ np.matrix([[0.0 for j in range(columnsMinusOne+1)] for i in range(rows)], dtype = np.float64) # +1 is for the bias units
                                for rows, columnsMinusOne in zip(self.layerSizes[:-1], self.layerSizes[1:])]    # Initialise the Theta arrays to zero
        # FIXME - what format should the training data be in? 


    def predictClass(self, inputVector):
        outputVector = self.__propForwardWholeNet__(inputVector)

