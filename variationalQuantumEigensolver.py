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

#if not IBMProvider.saved_accounts():
#    IBMProvider.save_account('3c3bfd24fb058fe3a5f07ca6eb998699052c9a6205e39754fb990663d1d7613fd0f25bdbd34f99691a40f4ced9931d3285a3a1c053a1fa1dfdfe0caeab34f04b')
#provider = IBMProvider()

def cvarExpectedValue(counts, shots, alpha):
    '''
    Calculates the Conditional Value at Risk of a Hamiltonian given the counts from a quantum circuit
    
    Parameters:
    counts (dict): A dictionary containing the counts from a quantum circuit
    shots (int): The number of shots
    alpha (float): The confidence level

    Returns:
    value (float): The Conditional Value at Risk of the Hamiltonian
    '''
    values = []

    for bitstring, count in counts.items():
        sign = (-1) ** bitstring.count('1')
        values.extend([sign] * count)

    values.sort()

    cutoffIndex = int(alpha * shots)

    cvarSum = sum(values[:cutoffIndex])

    cvarValue = cvarSum / cutoffIndex

    return cvarValue

def ParametricCircuitVQE(circ, n, theta):
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

def create_parametric_circuitVQE(n, nLayers):
    '''
    Creates a parameterized quantum circuit for the VQE algorithm
    
    Parameters:
    n (int): The number of qubits
    nLayers (int): The number of layers
    
    Returns:
    circ (QuantumCircuit): The parameterized quantum circuit
    theta_params (list): A list of parameter objects
    '''
    theta_params = [Parameter(f'theta_{i}') for i in range(n * nLayers)]
    circ = QuantumCircuit(n, n)
    counter = 0
    for layer in range(nLayers):
        for qubit in range(n):
            circ.ry(theta_params[counter], qubit)
            counter += 1
        for qubit in range(n - 1):
            circ.cx(qubit, qubit + 1)
    return circ, theta_params

def NParaVQE(n):
    '''
    Calculates the number of parameters needed for the VQE algorithm
    
    Parameters:
    n (int): The number of qubits
    
    Returns:
    n * k (int): The number of parameters
    k (int): The number of layers
    '''
    k = 1
    while n * k <= n ** 2:
        k += 1
    k -= 1
    return n * k, k

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
    pm = generate_preset_pass_manager(optimization_level=1, backend=backend)
    isa_circuit = pm.run(circ)

    sampler = Sampler(mode=session)

    sampler.options.default_shots = shots

    job = sampler.run([isa_circuit])

    result = job.result()
    pub_result = result[0]

    return pub_result.data.c.get_counts()

def cvarVQE(theta, quantumParameters, session, backend, parametric_circuit, theta_params):
    '''
    Calculates the Conditional Value at Risk of the Hamiltonian given the parameters theta
    
    Parameters:
    theta (list): A list of parameters
    quantumParameters (dict): A dictionary containing quantum parameters
    session (Session): The Qiskit runtime session
    backend (Backend): The backend to run the circuit on
    parametric_circuit (QuantumCircuit): The parameterized quantum circuit
    theta_params (list): A list of parameter objects
    
    Returns:
    ev (float): The Conditional Value at Risk of the Hamiltonian
    '''
    n = quantumParameters['nVQE']
    h = quantumParameters['hVQE']
    shots = quantumParameters['shots']
    alpha = quantumParameters['cvarAlpha']
    ev = 0
    for key in h:
        if key != '0' * n:
            bound_circ = parametric_circuit.assign_parameters({theta_params[i]: theta[i] for i in range(len(theta))})
            for j in range(n):
                if key[j] == '3':
                    bound_circ.measure(j, j)
            if quantumParameters['machine'] == 'simulator':
                counts = Sim(bound_circ, shots)
            elif quantumParameters['machine'] == 'real':
                counts = SimReal(bound_circ, shots)
            elif quantumParameters['machine'] == 'realSession':
                counts = SimRealSession(bound_circ, shots, session, backend)
            value = cvarExpectedValue(counts, shots, alpha)
            ev += h[key] * value
        else:
            ev += h[key]
    return ev

def thetaToSolutionVQE(theta, parameters):
    '''
    Converts the parameters theta to a solution for the VQE algorithm
    
    Parameters:
    theta (list): A list of parameters
    parameters (dict): A dictionary containing wind farm parameters
    
    Returns:
    solution (list): A list representing the wind farm layout solution
    '''
    circ = QuantumCircuit(parameters['nVQE'], parameters['nVQE'])
    circ = ParametricCircuitVQE(circ, parameters['nVQE'], parameters['nLayersVQE'], theta)
    for j in range(parameters['nVQE']):
        circ.x(j)
        circ.measure(j, j)
    counts = Sim(circ, parameters['shots'])

    bestKey = max(counts, key=counts.get)
    return [int(i) for i in bestKey]
