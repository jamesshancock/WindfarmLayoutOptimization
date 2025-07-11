# WindfarmLayoutOptimization

(WFLO)
All the code for WFLO (Jansen wake model) using quantum computing. The methods included are VQE-based, PCE-based, and exhaustive check.

===========================================================================

This project will require the following packages to run:
- numpy
- scipy
- qiskit
- qiskit_aer
- itertools
- matplotlib

===========================================================================

Project runs from main.py, controlled by a set of parameters. Below is a description of these parameters.

'len_grid': Integer
- Length of one side of the square windfarm grid.

'D': List(List(Float, Float, Float))
- Wind regime.
- Inner list format: [angle, speed, probability].

'x': Float
- Length back wake will cover.

'r': Float
- Radius of wake cone per unit back.

'm': Integer
- Maximum number of turbines.

'E': Float
- Proximity threshold.

'noGos': List(Tuple(Int, Int))
- List of disallowed grid positions for turbine placement.

'lam1': Float
- Weight for number constraint.
- Recommended: 7e3.

'lam2': Float
- Weight for proximity constraint.
- Recommended: 7e3.

'lam3': Float
- Weight for location constraint (e.g., no-go zones).
- Recommended: 1e3.

'shots': Integer
- Circuit calls per quantum expected value calculation.
- Recommended: >1000.

'machine': String
- Backend type.
- Options: 'simulator', 'real', 'realSession'.
- This setting currently does NOT work, requiring a qiskit license.

'solver': String
- Which quantum mapping to use.
- Options:
  - 'VQE': spin Hamiltonian form.
  - 'PCE': Pauli Correlation Encoding.
  - 'combinedPCE': PCE with grouped observable measurements.

'method': String
- Classical optimizer to use.
- Options:
  - For VQE or PCE: 'bayesian', 'COBYLA', or 'SLSQP'.
  - For efficientPCE: 'gradientDescent' or 'stochasticGradientDescent'.

'stepSize': Float
- Step size for gradient descent methods (efficientPCE only).
- Recommended: 0.1.

'learningRate': Float
- Learning rate for gradient descent (efficientPCE only).
- Recommended: 1 / (10 * lam1), e.g. 1/(10 * 7000).

'nSamples': Integer
- Number of complete runs to perform.

'talpha': Float
- Alpha hyperparameter for PCE step function.
- Recommended: 4.0.

'fixedK': List(Bool, Int)
- Whether to fix number of qubits (k) used in PCE.
- Example: [True, 1] to fix k = 1.
- Set to [False, _] to let k vary.

'L': Integer
- Number of previous costs used in the average stopping condition.
- Recommended: 3.

'tol': Float
- Tolerance on stopping condition (average cost change).
- Recommended: 0.1.

'miniter': Integer
- Minimum number of optimization iterations.
- Recommended: 10.

'maxiter': Integer
- Maximum number of optimization iterations.
- Recommended: 1000.

'bayesSigma': Float
- Sigma hyperparameter for Bayesian kernel.
- Recommended: 1000.0.
- See: https://arxiv.org/abs/2406.06150

'bayesGamma': Float
- Gamma hyperparameter for Bayesian kernel.
- Recommended: 0.01.
- See: https://arxiv.org/abs/2406.06150

'bayesIters': Integer
- Number of steps in Bayesian optimization.
- Recommended: 100.

'samples': Integer
- Number of output samples to collect.
- Usually same as 'nSamples'.

'cvarAlpha': Float
- Proportion of samples used in CVaR evaluation ('VQE' only).
- Range: 0.0 to 1.0.
- Recommended: 0.8.

'ExhaustiveCheck': Boolean
- Whether to perform an exhaustive check for the best spin configuration.
- Warning: Computationally expensive for large grids.
