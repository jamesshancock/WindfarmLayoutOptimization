# Optimizers: COByLA, Bayesian (NEW kernel !!)
'''
Packages
'''
from re import L
import numpy as np
import os
import sys
import concurrent.futures
import time
from scipy.optimize import minimize

'''
Modules
'''
from oldBayesian import BayesianOptimization
from windfarmQUBO import *
from variationalQuantumEigensolver import *
from extras import *

print("At least its working")

parameters = {
    'len_grid': 3,  # Grid side length
    'D': [[10 * k, 12, 1 / 36] for k in range(36)],  # Windregime
    'x': 0,  # Wake length
    'r': 0.0,  # Wake radius
    'm': 1,  # Number of turbines
    'E': 0.0,  # Proximity threshold
    'noGos': [],  # No-go zones
    'lam1': 7e3,  # Number constraint
    'lam2': 7e3,  # Proximity constraint
    'lam3': 1e3,  # Location constraint
    'shots': 300,  # Quantum shots
    'machine':
    'simulator',  # If you want to access real or realSession hardware - requires a new key
    'solver': 'combinedPCE',  # VQE: spin hamiltonian form
    # PCE: Pauli Correlation Encoding (PCE)
    # combinedPCE: PCE, but measures multiple operators when possible
    # efficientPCE: SQOE - single qubit operators, acting over N/2 qubits
    'method':
    'COBYLA',  # When using efficientPCE: gradientDescent or stochasticGradientDescent
    # When using PCE or VQE: bayesian, COBYLA or SLSQP
    'stepSize': 0.1,  # For efficienrPCE
    'learningRate': 1 / (10 * 7000),  # For efficientPCE
    'nSamples': 1,  # Number of samples to take
    'talpha': 4.0,  # Alpha inside step function
    'fixedK': [
        True, 1
    ],  # Set a fixed k for PCE, set first value to False to choose minimum qubits
    'L': 3,  # Number of costs stored to take average of for stopping criteria
    'tol': 1e-1,  # Tolerance on stopping criteria
    'miniter': 10,  # Minimum number of iterations
    'maxiter': 1000,  # Maximum number of iterations
    'bayesSigma': 1000.0,  # }
    'bayesGamma': 0.01,  # } Bayesian parameters for the new kernel
    'bayesIters': 100,  # }
    'samples': 1,  # }
    'cvarAlpha': 0.8,  # Alpha for CVaR
    'ExhaustiveCheck':
    False,  # If True, perform exhaustive check for optimal spin configuration
}

# TEST ZONE
# =============================================================================

Q = WindfarmQ(parameters)
parameters['nVQE'] = parameters['len_grid'] * parameters['len_grid']
parameters['nParaVQE'], parameters['nLayersVQE'] = NParaVQE(parameters['nVQE'])
h = QuboToVQE(Q, parameters)
parameters['hVQE'] = h

energies, histories, timeTakens, solutions = runForSamples(parameters)
print("energies=",energies)
print("histories=",histories)
print("timeTakens=",timeTakens)
print("solutions=",solutions)
