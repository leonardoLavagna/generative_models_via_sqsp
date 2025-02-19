from qiskit_aer import Aer
from scipy.stats import beta

# Quantum backend simulator
backend = Aer.get_backend("aer_simulator")
shots = 1024

# Number of qubits
m = 4

# Optimizer 
optimizer_type = "COBYLA"
max_iterations = 20
runs = 10

# Target distribution
max_trainable_params = 2**m-1
alpha_ = 2
beta_ = 2
a = 0
b = 1     
