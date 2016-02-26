#! /usr/bin/python2.7

import unittest

import numpy as np

from pyNeuralNet import pyNeuralNet

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

if __name__ == "__main__":
    unittest.main()
