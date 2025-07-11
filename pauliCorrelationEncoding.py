from qiskit import QuantumCircuit, transpile
#from qiskit_ibm_provider import IBMProvider
#from qiskit_aer.backends.aer_simulator import AerSimulator
from qiskit_aer import AerSimulator
from qiskit import transpile
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.compiler import transpile
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
import numpy as np
from extras import *
from qiskit.circuit import Parameter
from itertools import combinations

global simCounts
simCounts = 0


#if not IBMProvider.saved_accounts():
#    IBMProvider.save_account('3c3bfd24fb058fe3a5f07ca6eb998699052c9a6205e39754fb990663d1d7613fd0f25bdbd34f99691a40f4ced9931d3285a3a1c053a1fa1dfdfe0caeab34f04b')
#provider = IBMProvider()

def ExpectedValue(counts, shots):
    '''
    Calculates the expected value of a Hamiltonian given the counts from a quantum circuit
    
    Parameters:
    counts (dict): A dictionary containing the counts from a quantum circuit
    shots (int): The number of shots
    
    Returns:
    value (float): The expected value of the Hamiltonian
    '''
    n = len(list(counts.keys())[0])
    value = 0
    for key in counts:
        sign = 1
        for i in key:
            if i == '1':
                sign *= -1
        value += sign * counts[key] / shots
    return value

def ParametricCircuitPCE(circ, n, theta):
    counter = 0
    totalParas = len(theta)
    nLayers = totalParas / n

    for layer in range(int(nLayers/1)):
        for qubit in range(n):
            circ.ry(theta[counter], qubit)
            counter += 1
        for qubit in range(n - 1):
            circ.cx(qubit, qubit + 1)
    return circ

def create_parametric_circuitPCE(n, nPara, nLayers):
    theta_params = [Parameter(f'theta_{i}') for i in range(nPara)]  # Fewer parameters
    circ = QuantumCircuit(n,n)

    # Initial Layer: Pauli-rotated qubits for independent control


    counter = 0  # Track parameter index

    for _ in range(nLayers):
        # Add minimal entanglement

        for qubit in range(n):
            circ.ry(theta_params[counter], qubit) 
            counter += 1

        for qubit in range(n):
            circ.rx(theta_params[counter], qubit)  # Z rotation controls <Z>
            counter += 1

        for qubit in range(0, n-1):  # Use staggered CNOTs for better control
            circ.cx(qubit, qubit + 1)

    return circ, theta_params

'''def NParaPCE(n):
    k = 1
    while 3 * n * k <= n:
        k += 1
    k -= 1
    print("Here:", n, k)
    return 3 * n * k, k'''

def Sim(circ, shots, backend_name='ibm_fez'):
    '''
    Runs the quantum circuit on a local simulator configured to mimic an IBM Quantum backend - returns the counts
    
    Parameters:
    circ (QuantumCircuit): The quantum circuit
    shots (int): The number of shots
    backend_name (str): The name of the IBM Quantum backend to mimic (default is 'ibmq_qasm_simulator')
    
    Returns:
    counts (dict): A dictionary containing the counts from the quantum circuit
    '''
    # Get the IBM Quantum backend to mimic
    #ibm_backend = provider.get_backend(backend_name)
    # Configure the Aer simulator to mimic the IBM Quantum backend
    #backend = AerSimulator.from_backend(ibm_backend)

    backend = AerSimulator()

    # Transpile the circuit for the backend
    transpiled_circ = transpile(circ, backend, optimization_level=3)

    # Run the circuit on the backend
    job = backend.run(transpiled_circ, shots=shots)

    # Wait for the job to complete and get the result
    result = job.result()

    global simCounts
    simCounts += 1
    #print("Sim call number",simCounts)

    return result.get_counts(circ)

def SimReal(circ, shots):
    '''
    Runs the quantum circuit and returns measurement counts.

    Parameters:
    circ (QuantumCircuit): The quantum circuit
    shots (int): The number of shots

    Returns:
    counts (dict): Dictionary of measured bitstring frequencies
    '''

    # Load IBM Quantum service
    service = QiskitRuntimeService(channel="ibm_quantum",
                                   token='3c3bfd24fb058fe3a5f07ca6eb998699052c9a6205e39754fb990663d1d7613fd0f25bdbd34f99691a40f4ced9931d3285a3a1c053a1fa1dfdfe0caeab34f04b')

    backend = service.least_busy(simulator=False, operational=True)

    pm = generate_preset_pass_manager(optimization_level=1, backend=backend)
    isa_circuit = pm.run(circ)

    sampler = Sampler(mode=backend)

    sampler.options.default_shots = shots

    job = sampler.run([isa_circuit])

    result = job.result()
    pub_result = result[0]

    return pub_result.data.c.get_counts()

def SimRealSession(circ, shots, session, backend):
    '''
    Runs the quantum circuit and returns measurement counts within a session.

    Parameters:
    circ (QuantumCircuit): The quantum circuit
    shots (int): The number of shots
    session (Session): The Qiskit runtime session

    Returns:
    counts (dict): Dictionary of measured bitstring frequencies
    '''
    pm = generate_preset_pass_manager(optimization_level=3, backend=backend)
    isa_circuit = pm.run(circ)

    sampler = Sampler(mode=session)

    sampler.options.default_shots = shots

    job = sampler.run([isa_circuit])

    result = job.result()
    pub_result = result[0]

    simCalls += 1

    return pub_result.data.c.get_counts()

def ExpectedValueForKey(theta, quantumParameters, key, session, backend, parametric_circuit, theta_params):
    '''
    Calculates the expected value of the Hamiltonian given the parameters theta
    
    Parameters:
    theta (list): A list of parameters
    quantumParameters (dict): A dictionary containing quantum parameters
    
    Returns:
    ev (float): The expected value of the Hamiltonian
    '''
    n = quantumParameters['nPCE']
    nLayers = quantumParameters['nLayersPCE']
    shots = quantumParameters['shots']
    circ = parametric_circuit.assign_parameters({theta_params[i]: theta[i] for i in range(len(theta))})

    for j in range(n):
        if key[j] == '3':
            circ.measure(j, j)
        elif key[j] == '1':
            circ.h(j)
            circ.measure(j, j)
        elif key[j] == '2':
            circ.sdg(j)
            circ.h(j)
            circ.measure(j, j)
    if quantumParameters['machine'] == 'simulator':
        counts = Sim(circ, shots)
    elif quantumParameters['machine'] == 'real':
        counts = SimReal(circ, shots)
    elif quantumParameters['machine'] == 'realSession':
        counts = SimRealSession(circ, shots ,session, backend)

    return np.tanh(quantumParameters['talpha']*ExpectedValue(counts, shots))

def PCE(theta, parameters, keys, Jprime, hPCE, session, backend, parametric_circuit, theta_params):
    '''
    Calculates the PCE cost function
    
    Parameters:
    theta (list): A list of parameters
    parameters (dict): A dictionary containing parameters
    keys (list): A list of keys
    Jprime (numpy.ndarray): The upper triangular matrix
    
    Returns:
    cost (float): The cost of the PCE
    '''
    store = []
    cost = 0
    for key in keys:
        store.append(ExpectedValueForKey(theta, parameters, key, session, backend, parametric_circuit, theta_params))
    for i in range(Jprime.shape[0]):
        for j in range(i+1,Jprime.shape[0]):
            cost += Jprime[i,j]*store[i]*store[j]
        cost += hPCE[i]*store[i]
    return cost

def findCombinations(hKeys):
    n = len(hKeys[0])

    measureLocs = {}
    for key in hKeys:
        measureLocs[key] = [i for i, char in enumerate(key) if char != '0']


    compatables = {}
    for key in hKeys:
        locs = measureLocs[key]
        compatables[key] = []
        for keyTest in hKeys:
            if key != keyTest:
                locsTest = measureLocs[keyTest]
                if not any(abs(loc - locTest) <= 1 for loc in locs for locTest in locsTest):
                    compatables[key].append(keyTest)


    compatible_sets = []
    keys_left = set(hKeys)

    while keys_left:
        key = keys_left.pop()
        current_set = {key}
        to_check = set(compatables[key])
        while to_check:
            test_key = to_check.pop()
            if test_key in keys_left and all(test_key in compatables[existing_key] for existing_key in current_set):
                current_set.add(test_key)
                keys_left.remove(test_key)
                to_check.update(compatables[test_key])

        compatible_sets.append(current_set)


    combinedOps = []
    for set1 in compatible_sets:
        string = '0'*n
        for i in range(n):
            for op in set1:
                if op[i] != '0':
                    string = string[:i] + op[i] + string[i + 1:]
        combinedOps.append(string)
    return compatible_sets, combinedOps, measureLocs

def combinedEVs(theta, quantumParameters, shots, circ, theta_params, n, combinedOps, compatible_sets, measureLocs, session, backend):
    counter = 0
    termEvs = {}
    ev = 0
    for key in combinedOps:
        if key != '0' * n:
            bound_circ = circ.assign_parameters({theta_params[i]: theta[i] for i in range(len(theta_params))})
            for j in range(n):
                if key[j] == '1':
                    bound_circ.h(j)
                    bound_circ.measure(j, j)
                elif key[j] == '2':
                    bound_circ.sdg(j)
                    bound_circ.h(j)
                    bound_circ.measure(j, j)
                if key[j] == '3':
                    bound_circ.measure(j, j)
            if quantumParameters['machine'] == 'simulator':
                counts = Sim(bound_circ, shots)
            elif quantumParameters['machine'] == 'real':
                counts = SimReal(bound_circ, shots)
            elif quantumParameters['machine'] == 'realSession':
                counts = SimRealSession(bound_circ, shots, session, backend)
            set2 = compatible_sets[counter]
            counter += 1
            for key2 in set2:
                locs = measureLocs[key2]
                coef = 1
                termEv = 0
                for countsKey in counts:
                    count = counts[countsKey]
                    sign = 1
                    countsKey = countsKey[::-1]
                    for loc in locs:
                        if countsKey[loc] == '1':
                            sign *= -1
                    termEv += sign * count / shots
                termEvs[key2] = termEv
                ev += coef * termEv
    return termEvs

def combinedPCE(theta, parameters, keys, Jprime, hPCE, session, backend, circ, theta_params, combinedOps, compatible_sets, measureLocs):
    n = len(list(keys[0]))
    shots = parameters['shots']
    termEVs = combinedEVs(theta, parameters, shots, circ, theta_params, n, combinedOps, compatible_sets, measureLocs, session, backend)
    cost = 0
    store = [np.tanh(parameters['talpha']*termEVs[key]) for key in keys]
    for i in range(Jprime.shape[0]):
        for j in range(i+1, Jprime.shape[0]):
            cost += Jprime[i, j] * store[i] * store[j]
        cost += hPCE[i] * store[i]
    return cost


def thetaToSolutionPCE(theta, parameters, keys, session, backend, parametric_circuit, theta_params):
    '''
    Converts the parameters theta to a solution for the PCE algorithm
    
    Parameters:
    theta (list): A list of parameters
    parameters (dict): A dictionary containing wind farm parameters
    keys (list): A list of keys
    
    Returns:
    solution (list): A list representing the wind farm layout solution
    '''
    store = []
    parameters['machine'] = 'simulator'
    for key in keys:
        EV = ExpectedValueForKey(theta, parameters, key, session, backend, parametric_circuit, theta_params)
        if EV > 0:
            store.append(1)
        else:
            store.append(0)
    return store
