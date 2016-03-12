#! /usr/bin/python2.7

import unittest

from math import exp
import numpy as np


from pyNeuralNet import pyNeuralNet, sigmoid, sigmoidGradient, sigmoid_broadcast, sigmoidGradient_broadcast

class pyNeuralNetTestMethods(unittest.TestCase):

    def test_init(self):
        net = pyNeuralNet(5, [6], 7)
        self.assertEqual(net.inputSize, 5)
        self.assertEqual(net.numberOfLayers, 3)
        self.assertEqual(net.layerSizes, [5, 6, 7])
        self.assertEqual(len(net.Theta), 2)
        self.assertEqual(net.Theta[0].shape, (6, 6))
        self.assertTrue((net.Theta[0] == np.matrix([[0.0 for j in range(6)] for i in range(6)])).all())
        self.assertEqual(net.Theta[1].shape, (7, 7))
        self.assertTrue((net.Theta[1] == np.matrix([[0.0 for j in range(7)] for i in range(7)])).all())

        net = pyNeuralNet(15, [3, 3, 8], 10)
        self.assertEqual(net.inputSize, 15)
        self.assertEqual(net.numberOfLayers, 5)
        self.assertEqual(net.layerSizes, [15, 3, 3, 8, 10])
        self.assertEqual(len(net.Theta), 4)
        self.assertEqual(net.Theta[0].shape, (3, 16))
        self.assertTrue((net.Theta[0] == np.matrix([[0.0 for j in range(16)] for i in range(3)])).all())
        self.assertEqual(net.Theta[1].shape, (3, 4))
        self.assertTrue((net.Theta[1] == np.matrix([[0.0 for j in range(4)] for i in range(3)])).all())
        self.assertEqual(net.Theta[2].shape, (8, 4))
        self.assertTrue((net.Theta[2] == np.matrix([[0.0 for j in range(4)] for i in range(8)])).all())
        self.assertEqual(net.Theta[3].shape, (10, 9))
        self.assertTrue((net.Theta[3] == np.matrix([[0.0 for j in range(9)] for i in range(10)])).all())

        with self.assertRaises(TypeError):
            pyNeuralNet(-1, [5, 5], 3)
        with self.assertRaises(TypeError):
            pyNeuralNet(5, [0], 3)
        with self.assertRaises(TypeError):
            pyNeuralNet(5, [], 3)
        with self.assertRaises(TypeError):
            pyNeuralNet(5, [5], 0)

    def test_sigmoid(self):
        self.assertEqual(sigmoid(0.0), 0.5)
        self.assertGreaterEqual(sigmoid(50.0), (1.0 - 1e-20))
        self.assertLessEqual(sigmoid(-50.0), 1e-20)
        
    def test_sigmoidGradient(self):
        self.assertEqual(sigmoidGradient(0.0), 0.25)
        self.assertAlmostEqual(sigmoidGradient(50.0), 0.0)
        self.assertAlmostEqual(sigmoidGradient(-50.0), 0.0)

        epsilon = 1e-7
        def numGrad(x):
            return (sigmoid(x+epsilon) - sigmoid(x-epsilon))/(2.0*epsilon)

        self.assertAlmostEqual(sigmoidGradient(1.0), numGrad(1.0))
        self.assertAlmostEqual(sigmoidGradient(-5.0), numGrad(-5.0))

    def test_propForwardOneLayer(self):
        net = pyNeuralNet(5, [6], 7)
        self.assertEqual(len(net.__propForwardOneLayer__(1, [0, 0, 0, 0, 0])), 6)
        self.assertTrue((net.__propForwardOneLayer__(1, [0, 0, 0, 0, 0]) == np.matrix([0.5, 0.5, 0.5, 0.5, 0.5, 0.5]).T).all())
        self.assertEqual(len(net.__propForwardOneLayer__(2, [0, 0, 0, 0, 0, 0])), 7)
        self.assertTrue((net.__propForwardOneLayer__(2, [0, 0, 0, 0, 0, 0]) == np.matrix([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]).T).all())

        epsilon = 1e-12
        net.Theta[0] = np.matrix([[5, 0, 0, 2, 1, 0],
                                  [0, 1, 0, 0, 0, 1],
                                  [1, 0, 0, 0, 0, 0],
                                  [0, 1, 1, 0, 0, 0],
                                  [0, 0, 0, 0, 1, 0],
                                  [5, 0, 0, 0, 0, 2]])
        net.Theta[1] = np.matrix([[2, 0, 1, 1, 0, 0, 1],
                                  [0, 1, 4, 0, 0, 1, 0],
                                  [1, 0, 0, 1, 0, 0, 0],
                                  [0, 0, 0, 1, 0, 0, 0],
                                  [0, 2, 0, 0, 3, 0 ,4],
                                  [2, 0, 3, 0, 2, 0, 0],
                                  [1, 0, 0, 1, 0, 0, 1]])
        a2 = net.__propForwardOneLayer__(1, np.matrix([4, -2, 6, 3, -1]).T)
        self.assertLessEqual((a2 - np.matrix([1.0 / (1.0 + exp(-z)) for z in [20,3,1,2,3,3]]).T).max(), epsilon)
        self.assertGreaterEqual((a2 - np.matrix([1.0 / (1.0 + exp(-z)) for z in [20,2,1,2,3,3]]).T).min(), -epsilon)
        a3 = net.__propForwardOneLayer__(2, np.matrix([2, -2, 3, -3, 1, 4]).T)
        self.assertLessEqual((a3 - np.matrix([1.0 / (1.0 + exp(-z)) for z in [7, -5, 4, 3, 11, -10, 8]]).T).max(), epsilon)
        self.assertGreaterEqual((a3 - np.matrix([1.0 / (1.0 + exp(-z)) for z in [7, -5, 4, 3, 11, -10, 8]]).T).min(), -epsilon)

    def test_propForwardZToZ(self):
        net = pyNeuralNet(5, [6], 7)
        self.assertEqual(len(net.__propForwardZToZ__(2, [0, 0, 0, 0, 0, 0])), 7)
        self.assertTrue((net.__propForwardZToZ__(2, [0.9, 0.9, 0.9, 0.9, 0.9, 0.9]) == np.matrix([0, 0, 0, 0, 0, 0, 0]).T).all())
        with self.assertRaises(TypeError):
            net.__propForwardZToZ__(1, [0.9, 0.9, 0.9, 0.9, 0.9])
        with self.assertRaises(TypeError):
            net.__propForwardZToZ__(3, [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9])

        epsilon = 1e-12
        net.Theta[0] = np.matrix([[5, 0, 0, 2, 1, 0],
                                  [0, 1, 0, 0, 0, 1],
                                  [1, 0, 0, 0, 0, 0],
                                  [0, 1, 1, 0, 0, 0],
                                  [0, 0, 0, 0, 1, 0],
                                  [5, 0, 0, 0, 0, 2]])
        net.Theta[1] = np.matrix([[2, 0, 1, 1, 0, 0, 1],
                                  [0, 1, 4, 0, 0, 1, 0],
                                  [1, 0, 0, 1, 0, 0, 0],
                                  [0, 0, 0, 1, 0, 0, 0],
                                  [0, 2, 0, 0, 3, 0 ,4],
                                  [2, 0, 3, 0, 2, 0, 0],
                                  [1, 0, 0, 1, 0, 0, 1]])
        z2 = np.matrix([0.5, -0.1, 0.4, 0.3, -0.7, -0.23]).T
        a2 = np.vstack([[1], sigmoid_broadcast(z2)])
        z3 = net.Theta[1] * a2
        self.assertLessEqual((net.__propForwardZToZ__(2, z2) - z3).max(), epsilon)
        self.assertGreaterEqual((net.__propForwardZToZ__(2, z2) - z3).min(), -epsilon)
        
    def test_propForwardAToZ(self):
        net = pyNeuralNet(5, [6], 7)
        self.assertEqual(len(net.__propForwardAToZ__(2, [0, 0, 0, 0, 0, 0])), 7)
        self.assertTrue((net.__propForwardAToZ__(2, [0.9, 0.9, 0.9, 0.9, 0.9, 0.9]) == np.matrix([0, 0, 0, 0, 0, 0, 0]).T).all())
        with self.assertRaises(TypeError):
            net.__propForwardAToZ__(1, [0.9, 0.9, 0.9, 0.9, 0.9, 0.9])
        with self.assertRaises(TypeError):
            net.__propForwardAToZ__(3, [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9])
        
        epsilon = 1e-12
        net.Theta[0] = np.matrix([[5, 0, 0, 2, 1, 0],
                                  [0, 1, 0, 0, 0, 1],
                                  [1, 0, 0, 0, 0, 0],
                                  [0, 1, 1, 0, 0, 0],
                                  [0, 0, 0, 0, 1, 0],
                                  [5, 0, 0, 0, 0, 2]])
        net.Theta[1] = np.matrix([[2, 0, 1, 1, 0, 0, 1],
                                  [0, 1, 4, 0, 0, 1, 0],
                                  [1, 0, 0, 1, 0, 0, 0],
                                  [0, 0, 0, 1, 0, 0, 0],
                                  [0, 2, 0, 0, 3, 0 ,4],
                                  [2, 0, 3, 0, 2, 0, 0],
                                  [1, 0, 0, 1, 0, 0, 1]])
        z2 = np.matrix([0.5, -0.1, 0.4, 0.3, -0.7, -0.23]).T
        a2 = sigmoid_broadcast(z2)
        z3 = net.Theta[1] * np.vstack([[1.0], a2])
        self.assertLessEqual((net.__propForwardAToZ__(2, a2) - z3).max(), epsilon)
        self.assertGreaterEqual((net.__propForwardAToZ__(2, a2) - z3).min(), -epsilon)

    def test_propForwardZToA(self):
        net = pyNeuralNet(5, [6], 7)
        self.assertEqual(len(net.__propForwardZToA__(2, [0, 0, 0, 0, 0, 0])), 6)
        self.assertTrue((net.__propForwardZToA__(2, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) == np.matrix([0.5, 0.5, 0.5, 0.5, 0.5, 0.5]).T).all())
        with self.assertRaises(TypeError):
            net.__propForwardZToA__(1, [0.9, 0.9, 0.9, 0.9, 0.9, 0.9])  # z1 doesn't exist
        with self.assertRaises(TypeError):
            net.__propForwardZToA__(2, [0.9, 0.9, 0.9, 0.9, 0.9])  # z2 should have 6 elements
        with self.assertRaises(TypeError):
            net.__propForwardZToA__(4, [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]) # z4 doesn't exist
        
        epsilon = 1e-12
        net.Theta[0] = np.matrix([[5, 0, 0, 2, 1, 0],
                                  [0, 1, 0, 0, 0, 1],
                                  [1, 0, 0, 0, 0, 0],
                                  [0, 1, 1, 0, 0, 0],
                                  [0, 0, 0, 0, 1, 0],
                                  [5, 0, 0, 0, 0, 2]])
        net.Theta[1] = np.matrix([[2, 0, 1, 1, 0, 0, 1],
                                  [0, 1, 4, 0, 0, 1, 0],
                                  [1, 0, 0, 1, 0, 0, 0],
                                  [0, 0, 0, 1, 0, 0, 0],
                                  [0, 2, 0, 0, 3, 0 ,4],
                                  [2, 0, 3, 0, 2, 0, 0],
                                  [1, 0, 0, 1, 0, 0, 1]])
        z2 = np.matrix([0.5, -0.1, 0.4, 0.3, -0.7, -0.23]).T
        a2 = sigmoid_broadcast(z2)
        self.assertLessEqual((net.__propForwardZToA__(2, z2) - a2).max(), epsilon)
        self.assertGreaterEqual((net.__propForwardZToA__(2, z2) - a2).min(), -epsilon)

    def test_cost(self):
        net = pyNeuralNet(5, [6], 7)
        Theta1       = np.matrix([[5, 0, 0, 2, 1, 0],
                                  [0, 1, 0, 0, 0, 1],
                                  [1, 0, 0, 0, 0, 0],
                                  [0, 1, 1, 0, 0, 0],
                                  [0, 0, 0, 0, 1, 0],
                                  [5, 0, 0, 0, 0, 2]])
        Theta2       = np.matrix([[2, 0, 1, 1, 0, 0, 1],
                                  [0, 1, 4, 0, 0, 1, 0],
                                  [1, 0, 0, 1, 0, 0, 0],
                                  [0, 0, 0, 1, 0, 0, 0],
                                  [0, 2, 0, 0, 3, 0 ,4],
                                  [2, 0, 3, 0, 2, 0, 0],
                                  [1, 0, 0, 1, 0, 0, 1]])
        net.Theta[0] = Theta1
        net.Theta[1] = Theta2

        trainingPredictor = np.matrix([1.0, -3.2, 4.2, 0.4, 1.2])
        trainingResponse = np.matrix([2])

        cost = net.__cost__(trainingPredictor, trainingResponse);
        expectedh = np.matrix([[0.99034614503980067], [0.99451198042713096], [0.8495477739862124], [0.67503752737682365], [0.99816120792402985], [0.99261028906872006], [0.93879956476945892]])
        y = np.matrix([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]).T
        expectedCost = - (y.T * np.log(expectedh)) - (1.0 - y).T * np.log(1.0 - expectedh)

        print "expectedCost: {0}".format(expectedCost)
        self.assertAlmostEqual(expectedCost, cost)
        
    def test_computeGradients(self):
        net = pyNeuralNet(5, [6], 7)
        Theta1       = np.matrix([[5, 0, 0, 2, 1, 0],
                                  [0, 1, 0, 0, 0, 1],
                                  [1, 0, 0, 0, 0, 0],
                                  [0, 1, 1, 0, 0, 0],
                                  [0, 0, 0, 0, 1, 0],
                                  [5, 0, 0, 0, 0, 2]], dtype='float64')
        Theta2       = np.matrix([[2, 0, 1, 1, 0, 0, 1],
                                  [0, 1, 4, 0, 0, 1, 0],
                                  [1, 0, 0, 1, 0, 0, 0],
                                  [0, 0, 0, 1, 0, 0, 0],
                                  [0, 2, 0, 0, 3, 0 ,4],
                                  [2, 0, 3, 0, 2, 0, 0],
                                  [1, 0, 0, 1, 0, 0, 1]], dtype='float64')
        net.Theta[0] = Theta1
        net.Theta[1] = Theta2
        #trainingPredictors = np.matrix([[1.0, -3.2, 4.2, 0.4, 1.2],
        #                               [0.0, 0.4, -1.3, 0.0, 0.0],
        #                               [-0.4, 1.0, -0.1, 0.0, 1.2],
        #                               [3.2, 0.4, 0.0, 0.4, -0.4]])
        #trainingResponses = np.matrix([1, 2, 1, 3]).T
        trainingPredictors = np.matrix([[1.0, -3.2, 4.2, 0.4, 1.2]])
        trainingResponses = np.matrix([1]).T

        grad = net.__computeGradients__(trainingPredictors, trainingResponses)

        numGradients = sum([theta.size for theta in net.Theta])
        # Numerically compute the gradient from the cost function
        epsilon = 1e-6
        numericalGrad = np.matrix([0.0 for i in range(numGradients)]).T
        for i in range(numGradients):
            net.Theta[0] = Theta1
            net.Theta[1] = Theta2
            
            if (i < net.Theta[0].size):
                net.Theta[0].flat[i] = net.Theta[0].flat[i] + epsilon
            else:
                j = i - net.Theta[0].size
                net.Theta[1].flat[j] = net.Theta[1].flat[j] + epsilon
            costPlus = net.__cost__(trainingPredictors, trainingResponses)
            
            if (i < net.Theta[0].size):
                net.Theta[0].flat[i] = net.Theta[0].flat[i] - 2.0*epsilon
            else:
                j = i - net.Theta[0].size
                net.Theta[1].flat[j] = net.Theta[1].flat[j] - 2.0*epsilon
            costMinus = net.__cost__(trainingPredictors, trainingResponses)
            
            numericalGrad[i] = (costPlus - costMinus)/(2.0*epsilon)

        epsilon2 = 1e-4
        print grad[0,0]
        print numericalGrad[0,0]
        print (grad[0,0]/numericalGrad[0,0])
        print (grad[1,0]/numericalGrad[1,0])
        print (grad[2,0]/numericalGrad[2,0])
        self.assertLessEqual((grad - numericalGrad).max(), epsilon2)
        self.assertGreaterEqual((grad - numericalGrad).min(), -epsilon2)
                
            

        

if __name__ == "__main__":
    unittest.main()
