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
def vqeSolver(parameters):
    parameters['nVQE'] = parameters['len_grid'] * parameters['len_grid']
    parameters['nParaVQE'], parameters['nLayersVQE'] = NParaVQE(parameters['nVQE'])
    h = QuboToVQE(WindfarmQ(parameters), parameters)
    parameters['hVQE'] = h
    #print("Number of qubits required:", parameters['nVQE'])

    # Create the parameterized circuit
    parametric_circuit, theta_params = create_parametric_circuitVQE(parameters['nVQE'], parameters['nLayersVQE'])
    theta = [np.pi * np.random.uniform(0, 2) for _ in range(parameters['nParaVQE'])]

    history = []
    L = parameters.get('L', 10)  # Default value for L if not provided
    tol = parameters.get('tol', 1e-4)  # Default tolerance if not provided
    paras = []
    timeTaken = 0

    backend = AerSimulator()

    session = 'hello'

    def cost_function(theta, *args, **kwargs):
        value = cvarVQE(theta, parameters, session, backend, parametric_circuit, theta_params)
        return value

    timePerIter = []
    start = [time.perf_counter()]  # Use a list for mutability

    def callback(xk):
        end = time.perf_counter()
        timePerIter.append(end - start[0])  # Time since last callback (i.e., last iteration)
        start[0] = end  # Update for next iteration
        print("Time taken for iter", len(history), ":", timePerIter[-1])

        parameters['talpha'] *= 1e4
        theta = xk
        value = cost_function(theta)
        history.append(value)
        parameters['talpha'] /= 1e4
        paras.append(theta)
        if len(history) >= L:
            lowestLTerms = sorted(history)[:L]
            avRecents = sum(lowestLTerms)/len(lowestLTerms)
            last = history[-1]
            avg_change = abs(avRecents - last)
            if avg_change <= tol and len(history) > parameters['miniter']:
                raise StopIteration("Stopping criterion met: average change <= tol")


    tok = time.perf_counter()
    try:
        mini = minimize(cost_function, theta, method='COBYLA', options={'maxiter': parameters['maxiter']}, callback=callback)
    except StopIteration as e:
        print(e)
        mini = type('obj', (object,), {'x': theta})
    tik = time.perf_counter()
    print("Total iterations: ", len(history))
    finalParas = paras[np.argmin(history)]
    solution = thetaToSolutionVQE(finalParas, parameters)
    timeTaken = tik - tok
    return solution, history, timeTaken, timePerIter

TimesPerItersCOBYLA = []
TimesPerItersBayes = []
print("Code running")

Q = WindfarmQ(parameters)
parameters['nVQE'] = parameters['len_grid'] * parameters['len_grid']
parameters['nParaVQE'], parameters['nLayersVQE'] = NParaVQE(parameters['nVQE'])
h = QuboToVQE(Q, parameters)
parameters['hVQE'] = h

num_cores = concurrent.futures.ProcessPoolExecutor()._max_workers
total_samples = num_cores
print("Number of cores available:", num_cores)

if __name__ == "__main__":
    print("In here")

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(BayesianOptimization, parameters) for _ in range(total_samples)]
        for future in concurrent.futures.as_completed(futures):
            solution, bestX, timePerIter = future.result()
            TimesPerItersBayes.append(timePerIter)

    print("Bayes time per iter:", TimesPerItersBayes)
    print("COBYLA time per iter:", TimesPerItersCOBYLA)
