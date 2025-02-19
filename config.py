from qiskit_aer import Aer


backend = Aer.get_backend("aer_simulator")
m = 4
shots = 1024
optimizer_type = "COBYLA"
max_iterations = 20
runs = 10
max_trainable_params = 2**m-1
