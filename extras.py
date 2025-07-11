import numpy as np
import itertools
import math
import matplotlib.pyplot as plt
from qiskit import *
from qiskit_aer import *
import sys

class Logger:
    def __init__(self, file_path):
        self.terminal = sys.stdout
        self.log = open(file_path, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


def QuboToIsing(Q):
    """
    Convert a QUBO matrix to Ising model parameters.
    
    Parameters:
    Q (numpy.ndarray): The QUBO matrix.
    
    Returns:
    h (numpy.ndarray): Linear coefficients for the Ising model.
    J (numpy.ndarray): Quadratic coefficients for the Ising model.
    constant (float): The constant shift in energy from the transformation.
    """
    Q = np.array(Q)
    n = Q.shape[0]

    h = np.zeros(n)
    J = np.zeros((n, n))

    constant = np.sum(Q) / 4

    for i in range(n):
        h[i] = 1/2*(np.sum(Q[i, :]))
        for j in range(i+1, n):
            J[i, j] = Q[i, j] /2
    return h, J, constant

def ConvertToUpperTriangular(J):
    """
    Convert a symmetric matrix J to an upper triangular matrix J' such that
    J'[i, j] = J[i, j] + J[j, i] for i < j.
    
    Parameters:
    J (numpy.ndarray): The symmetric matrix.
    
    Returns:
    J_prime (numpy.ndarray): The upper triangular matrix.
    """
    J = np.array(J)

    n = J.shape[0]

    J_prime = np.zeros_like(J)

    for i in range(n):
        for j in range(i + 1, n):
            J_prime[i, j] = J[i, j] + J[j, i]

    return J_prime

def QuboToVQE(Q, parameters):
    '''
    Finds the Hamiltonian for the VQE algorithm from the windfarm parameters
    
    Parameters:
    parameters (dict): A dictionary containing wind farm parameters.
    
    Returns:
    h (dict): Linear coefficients for the Ising model.
    '''
    linearTerms, quadraticTerms, constant = QuboToIsing(Q)
    Jprime = ConvertToUpperTriangular(quadraticTerms)
    h = {}
    string = '0'*parameters['nVQE']
    stringlist = list(string)
    for i in range(parameters['nVQE']):
        for j in range(i+1,parameters['nVQE']):
            if i != j:
                term = stringlist.copy()
                term[i] = '3'
                term[j] = '3'
                h[''.join(term)] = Jprime[i,j]
        term = stringlist.copy()
        term[i] = '3'
        h[''.join(term)] = linearTerms[i]

    return h

def chooseNfixedK(len_grid, k):
    '''
    Finds the smallest n such that n choose k is greater than or equal to len_grid**2
    
    Parameters:
    len_grid (int): The number of grid points in one dimension
    k (int): The number of 1s, 2s, and 3s in the binary strings
    
    Returns:
    n (int): The smallest n such that n choose k is greater than or equal to len_grid**2
    '''
    N = len_grid ** 2
    n = k  # Start from k since n choose k is not defined for n < k

    while True:
        if 3*math.comb(n, k) >= N:
            return n
        n += 1


def chooseNandK(len_grid):
    '''
    Finds the smallest n and k such that n choose k is greater than or equal to len_grid**2
    
    Parameters:
    len_grid (int): The number of grid points in one dimension
    
    Returns:
    n (int): The smallest n such that n choose k is greater than or equal to len_grid**2
    k (int): The smallest k such that n choose k is greater than or equal to len_grid**2
    '''
    N = len_grid ** 2
    n = 1

    while True:
        for k in range(n + 1):
            if 3 * math.comb(n, k) >= N:
                return n, k
        n += 1

def generateBinaryStrings(n, k):
    '''
    Generates all binary strings of length n with exactly k 1s
    
    Parameters:
    n (int): The length of the binary strings
    k (int): The number of 1s in the binary strings
    
    Returns:
    list: A list of binary strings
    '''
    result = []
    for positions in itertools.combinations(range(n), k):
        binaryString = ['0'] * n
        for pos in positions:
            binaryString[pos] = '1'
        result.append(''.join(binaryString))
    return result

def generateThreeStrings(n, k):
    '''
    Generates all binary strings of length n with exactly k 3s
    
    Parameters:
    n (int): The length of the binary strings
    k (int): The number of 3s in the binary strings
    
    Returns:
    list: A list of binary strings
    '''
    result = []
    for positions in itertools.combinations(range(n), k):
        binaryString = ['0'] * n
        for pos in positions:
            binaryString[pos] = '3'
        result.append(''.join(binaryString))
    return result

def generateBinaryStringsWithCopies(n, k):
    '''
    Generates all binary strings of length n with exactly k 1s, 2s, and 3s
    
    Parameters:
    n (int): The length of the binary strings
    k (int): The number of 1s, 2s, and 3s in the binary strings
    
    Returns:
    list: A list of binary strings with 1s, 2s, and 3s
    '''
    result = []
    for positions in itertools.combinations(range(n), k):
        for digit in ['3', '2', '1']:
            binaryString = ['0'] * n
            for pos in positions:
                binaryString[pos] = digit
            result.append(''.join(binaryString))
    return result

def generateAllBinaryStrings(n):
    '''
    Generates all binary strings of length n
    
    Parameters:
    n (int): The length of the binary strings
    
    Returns:
    list: A list of binary strings
    '''
    result = []
    for k in range(1,n + 1):
        result += generateThreeStrings(n, k)

    return result

def exhaustiveCheck(Q, parameters):
    '''
    Checks all possible wind farm layouts and returns the optimal one.
    
    Parameters:
    parameters (dict): A dictionary containing wind farm parameters.
    
    Returns:
    optimal (list): The optimal wind farm layout.
    '''
    keys = []
    for k in range(parameters['len_grid']**2):
        keys += generateBinaryStrings(parameters['len_grid'] ** 2, k)
    keys.append('1'*parameters['len_grid']**2)
    optimal = None
    minEnergy = np.inf
    for key in keys:
        vector = np.array([int(i) for i in key])
        energy = vector.T @ Q @ vector
        if energy < minEnergy:
            minEnergy = energy
            optimal = vector
    return optimal

def spinExhaustiveCheck(Q, parameters):
    '''
    Checks all possible wind farm layouts and returns the optimal one as a spin cost.
    
    Parameters:
    parameters (dict): A dictionary containing wind farm parameters.
    
    Returns:
    minEnergy (float): The minimum energy of the optimal configuration.
    optimal (list): The optimal wind farm layout in spin format.
    '''
    keys = []
    for k in range(parameters['len_grid']**2):
        keys += generateBinaryStrings(parameters['len_grid'] ** 2, k)
    keys.append('1'*parameters['len_grid']**2)

    optimal = None
    minEnergy = np.inf

    hPCE, J, _ = QuboToIsing(Q)

    for key in keys:
        vector = np.array([int(i) for i in key])
        spinVector = 2 * vector - 1
        energy = spinVector.T @ J @ spinVector + hPCE.T @ spinVector
        if energy < minEnergy:
            minEnergy = energy
            optimal = spinVector
    print("The optimal spin energy is", minEnergy)
    print("One optimal solution is", optimal)
    return minEnergy, optimal

def solutionToGrid(solution, parameters):
    '''
    Converts a solution to a grid
    
    Parameters:
    solution (list): A list representing the wind farm layout solution
    parameters (dict): A dictionary containing wind farm parameters
    
    Returns:
    solutionGrid (numpy.ndarray): A grid representing the wind farm layout solution
    '''
    counter = 0
    solutionGrid = np.zeros((parameters['len_grid'], parameters['len_grid']))
    for i in range(parameters['len_grid']):
        for j in range(parameters['len_grid']):
            solutionGrid[j, i] = solution[counter]
            counter += 1
    return solutionGrid

def plot_histories(histories, timeTakens, title, method, xlabel, ylabel, text_offset=0.1):
    """
    Plots the histories with multiple lines and marks where each algorithm stopped, labeling with the time taken.

    Parameters:
    histories (list of lists): List of lists where each sublist is a history for one run.
    timeTakens (list of floats): List of time taken for each run.
    title (str): The title of the plot.
    xlabel (str): The label for the x-axis.
    ylabel (str): The label for the y-axis.
    text_offset (float): The vertical offset for the time labels.
    """
    plt.figure(figsize=(12, 8), dpi=150)

    for i, (history, timeTaken) in enumerate(zip(histories, timeTakens)):
        line, = plt.plot(history, label=f'Run {i+1}')
        # Mark the stopping point with the same color as the line
        plt.plot(len(history) - 1, history[-1], 'o', color=line.get_color())
        # Label the stopping point with the time taken, with a larger gap above the point
        #plt.text(len(history) - 1, history[-1] + text_offset, f'{timeTaken:.2f}s', fontsize=9, color=line.get_color(), ha='center')
    Title = title + " using "+ method
    plt.title(Title, fontsize=12)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.legend()
    plt.grid(True)

    plt.show()

    return None

def NoiseTester():
    def noiseTester(n, shots):
        Npara, Nlayers = NParaPCE(n)
        theta = np.random.rand(Npara)
        samples = list(range(100))
        se7 = []
        print("=====================================")
        print("Noise Tester")
        print("n:", n)
        print("shots:", shots)
        for _ in samples:
            q = QuantumRegister(n)
            c = ClassicalRegister(n)
            circ = QuantumCircuit(q,c)
            circ = ParametricCircuitPCE(circ, n, Nlayers, theta)
            circ.measure_all()
            counts = Sim(circ, shots)
            se7.append(ExpectedValue(counts, shots))
        sigma = np.std(se7)
        epsilon = sigma * np.sqrt(3)
        print("Sigma:", sigma)
        print("Epsilon:", epsilon)
        return epsilon

    noiselevel = [100,200,500,1000,2000,3000,4000,10000,50000]
    for noise in noiselevel:
        noiseTester(3, noise)
        print("=====================================")

def moving_average(data, window_size):
    """
    Calculate the moving average of the given data.
    
    Parameters:
    data (list): The input data.
    window_size (int): The size of the moving window.
    
    Returns:
    list: The smoothed data.
    """
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')
