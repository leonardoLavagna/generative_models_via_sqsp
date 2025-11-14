from qiskit_aer import Aer
from scipy.stats import beta

# Quantum backend simulator
backend = Aer.get_backend("aer_simulator")
shots = 1024

# Number of qubits and trainable parameters
m = 6
max_trainable_params = 2**m-1

# Optimizer 
optimizer_type = "COBYLA"
max_iterations = 20
runs = 10

# Target distribution
alpha_ = 2
beta_ = 2
a = 0
b = 1     
