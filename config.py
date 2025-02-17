from qiskit_aer import Aer


backend = Aer.get_backend("aer_simulator")
m = 3
shots = 1024
optimizer_type = "COBYLA"
max_iterations = 20
