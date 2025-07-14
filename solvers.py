from gettext import find
from scipy.optimize import minimize
import numpy as np
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Kernel, Hyperparameter
import concurrent.futures
import time
from qiskit_ibm_runtime import Session, QiskitRuntimeService

from extras import *
from windfarmQUBO import *
from variationalQuantumEigensolver import *
from pauliCorrelationEncoding import *
from efficientSolver import *
from oldBayesian import *

def NParaPCE(n):
    k = 1
    while 2 * n * k <= n:
        k += 1
    #k -= 1
    print("Here")
    print("Number of parameters:", 2 * n * k)
    print("Number of layers:", k)
    return 2 * n * k, k

def printFaff(parameters):
    '''
    Prints the optimal energy and solution for the wind farm layout optimization problem.
    
    Parameters:
    parameters (dict): A dictionary containing wind farm parameters.
    
    Returns:
    optEnergy (float): The optimal energy of the wind farm layout
    '''
    Q = WindfarmQ(parameters)
    if parameters['ExhaustiveCheck']:
        solution = exhaustiveCheck(Q, parameters)
        optEnergy = Energy(parameters, solution)
    print("----------------------------")
    print("Windfarm Layout Optimization")
    print("----------------------------")
    print("This system uses the following restrictions:")
    print("No more than", parameters['m'], "turbines")
    print("With a minimum spacing of", parameters['E'])
    print("Locations to avoid:")
    matr = np.zeros((parameters['len_grid'],parameters['len_grid']))
    counter = 1
    for j in range(parameters['len_grid']):
        for i in range(parameters['len_grid']):
            if counter in parameters['noGos']:
                matr[i,j] = 1
            counter += 1
    print(matr)
    if parameters['ExhaustiveCheck']:
        print("The optimal energy is", optEnergy)
        print("One optimal solution is", solution)
        print("As a grid:")
        print(solutionToGrid(solution, parameters))
    return 'done'

def quantumSolver(parameters):
    '''
    Solves the wind farm layout optimization problem using the specified quantum algorithm (VQE or PCE)
    
    Parameters:
    parameters (dict): A dictionary containing wind farm parameters
    
    Returns:
    solution (list): A list representing the wind farm layout
    '''
    if parameters['solver'] == 'VQE':
        parameters['nVQE'] = parameters['len_grid'] * parameters['len_grid']
        parameters['nParaVQE'], parameters['nLayersVQE'] = NParaVQE(parameters['nVQE'])
        h = QuboToVQE(WindfarmQ(parameters), parameters)
        parameters['hVQE'] = h
        print("Number of qubits required:", parameters['nVQE'])

        # Create the parameterized circuit
        parametric_circuit, theta_params = create_parametric_circuitVQE(parameters['nVQE'], parameters['nLayersVQE'])
        theta = [np.pi * np.random.uniform(0, 2) for _ in range(parameters['nParaVQE'])]
    elif parameters['solver'] == 'PCE' or 'combinedPCE':

        if parameters['fixedK'][0]:
            parameters['nPCE'] = chooseNfixedK(parameters['len_grid'], parameters['fixedK'][1])
            parameters['kPCE'] = parameters['fixedK'][1]
        else:
            parameters['nPCE'], parameters['kPCE'] = chooseNandK(parameters['len_grid'])
        parameters['nParaPCE'], parameters['nLayersPCE'] = NParaPCE(parameters['nPCE'])
        keys = generateBinaryStringsWithCopies(parameters['nPCE'], parameters['kPCE'])

        keys = keys[:parameters['len_grid'] * parameters['len_grid']]

        hPCE, Jprime, _ = QuboToIsing(WindfarmQ(parameters))
        print("Number of qubits required:", parameters['nPCE'])
        print("PCE variables:")
        print(keys)
        theta = [np.pi * np.random.uniform(0, 2) for _ in range(parameters['nParaPCE'])]
        # Create the parameterized circuit
        parametric_circuit, theta_params = create_parametric_circuitPCE(parameters['nPCE'], parameters['nParaPCE'], parameters['nLayersPCE'])
        simsPerRun = len(keys)
        if parameters['solver'] == 'combinedPCE':
            compatible_sets, combinedOps, measureLocs = findCombinations(keys)
            simsPerRun = len(combinedOps)

    history = []
    L = parameters.get('L', 10)  # Default value for L if not provided
    tol = parameters.get('tol', 1e-4)  # Default tolerance if not provided
    paras = []
    timeTaken = 0


    # Initialize the Qiskit Runtime service
    #service = QiskitRuntimeService(channel="ibm_quantum",
    #token='3c3bfd24fb058fe3a5f07ca6eb998699052c9a6205e39754fb990663d1d7613fd0f25bdbd34f99691a40f4ced9931d3285a3a1c053a1fa1dfdfe0caeab34f04b')
    if parameters['machine'] == 'realSession':
        print("Error: realSession is not supported.")
    else:
        backend = AerSimulator()

        session = 'hello'

    def cost_function(theta, *args, **kwargs):
        if parameters['solver'] == 'VQE':
            value = cvarVQE(theta, parameters, session, backend, parametric_circuit, theta_params)
        elif parameters['solver'] == 'PCE':
            #value = PCE(theta, parameters, keys, Jprime, hPCE, session, backend, parametric_circuit, theta_params)
            value = PCE(theta, parameters, keys, Jprime, hPCE, session,
                        backend, parametric_circuit, theta_params)
        elif parameters['solver'] == 'combinedPCE':
            value = combinedPCE(theta, parameters, keys, Jprime, hPCE, session, backend, parametric_circuit, theta_params, combinedOps, compatible_sets, measureLocs)
        return value

    def callback(xk):
        parameters['talpha'] *= 1e4
        if parameters['method'] == 'COBYLA' or parameters['method'] == 'SLSQP':
            theta = xk
            value = cost_function(theta)
        elif parameters['method'] == 'Bayesian':
            theta = xk.x
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
    if parameters['method'] == 'COBYLA' or parameters['method'] == 'SLSQP':
        try:
            mini = minimize(cost_function, theta, method=parameters['method'], options={'maxiter': parameters['maxiter']}, callback=callback)
        except StopIteration as e:
            print(e)
            mini = type('obj', (object,), {'x': theta})
    elif parameters['method'] == 'Bayesian':
        solution, xbests, _, _ = BayesianOptimization(parameters)
        history = [np.array(s).T @ Q @ np.array(s) for s in xbests]
        
    tik = time.perf_counter()
    print("Total iterations: ", len(history))
    finalParas = paras[np.argmin(history)]
    if parameters['solver'] == 'VQE' and parameters['method'] != 'Bayesian':
        solution = thetaToSolutionVQE(finalParas, parameters)
    elif parameters['solver'] == 'PCE' or 'combinedPCE' and parameters['method'] != 'Bayesian':
        solution = thetaToSolutionPCE(finalParas, parameters, keys, session, backend, parametric_circuit, theta_params)
    print("Final cost: ", min(history))
    print("Total circuit calls:", len(history)*simsPerRun)
    timeTaken = tik - tok
    return solution, history, timeTaken

def runForSamples(parameters):
    '''
    Runs the quantum solver for a number of samples
    
    Parameters:
    parameters (dict): A dictionary containing quantum parameters
    nSamples (int): The number of samples
    
    Returns:
    energies (list): A list of energies
    '''
    print("Solver:", parameters['solver'])
    print("Method:", parameters['method'])
    print("Machine:",parameters['machine'])
    energies = []
    histories = []
    timeTakens = []
    solutions = []
    for _ in range(parameters['nSamples']):
        print("-------------------------------------------")
        if parameters['solver'] == 'efficientPCE':
            solution, history, timeTaken = optimizedSolver(parameters)
        else:
            solution, history, timeTaken = quantumSolver(parameters)
        count = 0
        if True:
            print("Total iterations: ", len(history))
            energy = Energy(parameters, solution)
            print("Energy: ", energy)
            print("Solution: ", solution)
            print("Time taken:", timeTaken, "seconds")
            print(solutionToGrid(solution, parameters))
            energies.append(Energy(parameters, solution))
            histories.append(history)
            timeTakens.append(timeTaken)
            solutions.append(solution)
    return energies, histories, timeTakens, solutions

def run_quantum_solver(parameters):
    print("Running")
    if parameters['solver'] == 'efficientPCE':
        solution, history, timeTaken = optimizedSolver(parameters)
    else:
        solution, history, timeTaken = quantumSolver(parameters)
    energy = Energy(parameters, solution)
    return energy, history, timeTaken, solution

def runForSamplesParallel(parameters):
    '''
    Runs the quantum solver for a number of samples in parallel
    
    Parameters:
    parameters (dict): A dictionary containing quantum parameters
    nSamples (int): The number of samples per core
    
    Returns:
    energies (list): A list of energies
    histories (list): A list of histories
    timeTakens (list): A list of time taken for each run
    solutions (list): A list of solutions
    '''
    print("Solver:", parameters['solver'])
    print("Method:", parameters['method'])
    energies = []
    histories = []
    timeTakens = []
    solutions = []

    # Determine the number of available cores
    num_cores = concurrent.futures.ProcessPoolExecutor()._max_workers

    # Total number of tasks to run
    #total_samples = num_cores * parameters['nSamples']
    total_samples = num_cores

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(run_quantum_solver, parameters) for _ in range(total_samples)]
        for future in concurrent.futures.as_completed(futures):
            energy, history, timeTaken, solution = future.result()
            print("Energy: ", energy)
            print("Solution: ", solution)
            print("Time taken:", timeTaken, "seconds")
            print(solutionToGrid(solution, parameters))
            energies.append(energy)
            histories.append(history)
            timeTakens.append(timeTaken)
            solutions.append(solution)

    return energies, histories, timeTakens, solutions

def vqeSolver2(parameters):
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
print("Code running")

