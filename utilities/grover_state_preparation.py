#------------------------------------------------------------------------------
# grover_state_preparation.py
#
# This module provides utility functions for quantum state preparation using 
# Grover's algorithm. It includes functions to compute the required rotation 
# angles and construct a quantum circuit implementing the state preparation.
#
# Functions included:
# - get_grover_angles(p_i_set, m): Computes the rotation angles required for 
#   Grover's state preparation based on a given probability distribution.
# - state_expansion(m, thetas): Constructs a quantum circuit that applies the 
#   rotations based on the calculated angles.
#
# These utilities are useful for quantum algorithms requiring efficient state 
# preparation, including generative models and amplitude encoding.
#
# References:
# - https://arxiv.org/pdf/quant-ph/0208112
#
# © Leonardo Lavagna 2025
# @ NESYA https://github.com/NesyaLab
#------------------------------------------------------------------------------


import numpy as np
from itertools import product
from qiskit.circuit.library import RYGate
from qiskit import QuantumCircuit, QuantumRegister


def get_grover_angles(p_i_set, m):
    """Calculates Grover's angles for a given set of probabilities.

    Args:
        p_i_set (list or array-like): A list of probabilities corresponding to quantum states.
        m (int): The number of qubits used in the quantum circuit.

    Returns:
        list: A list of rotation angles required for Grover's state preparation.

    Raises:
        ValueError: If the computed number of angles does not match the required length for m qubits.
    """
    thetas = []
    p_i_set = list(p_i_set)
    for j in range(m):
        while len(p_i_set) % 2**(j + 1) != 0:
            p_i_set.append(0)
        current_p = np.array(np.array_split(p_i_set, 2**(j + 1))).sum(axis=1)
        if j > 0:
            for i in range(len(previous_p)):
                if previous_p[i] == 0:
                    previous_p[i] = 1e-5
                theta = 2 * np.arccos(np.sqrt(current_p[i * 2] / previous_p[i]))
                thetas.append(theta)
        else:
            theta = 2 * np.arccos(np.sqrt(current_p[0]))
            thetas.append(theta)
        previous_p = current_p
    required_length = 2**m - 1
    if len(thetas) != required_length:
        raise ValueError(f"The number of angles in 'thetas' must be {required_length} for {m} qubits.")
    return thetas


def state_expansion(m, thetas):
    """Constructs a quantum circuit that applies rotations based on the calculated angles.
    
    Args:
        m (int): The number of qubits in the circuit.
        thetas (list): A list of rotation angles computed for Grover's algorithm.
    
    Returns:
        QuantumCircuit: A quantum circuit implementing the state preparation.
    
    Raises:
        ValueError: If the number of angles does not match the required 2^m - 1.
    """
    if len(thetas) != 2**m - 1:
        raise ValueError("The number of angles in 'thetas' must be 2^m - 1 for m qubits.")
    qc = QuantumCircuit(m, m)
    qc.ry(thetas[0], 0)
    theta_index = 1
    for qubit in range(1, m):
        control_combinations = list(product([0, 1], repeat=qubit))
        for combination in control_combinations:
            for ctrl, state in enumerate(combination):
                if state == 0:
                    qc.x(ctrl)
            qc.append(RYGate(thetas[theta_index]).control(len(combination)), list(range(qubit)) + [qubit])
            theta_index += 1
            for ctrl, state in enumerate(combination):
                if state == 0:
                    qc.x(ctrl)
    qc.measure(range(m), range(m)[::-1])
    return qc
