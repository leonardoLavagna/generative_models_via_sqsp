#------------------------------------------------------------------------------
# compiler.py
#
# This module provides functions for compiling operators into a specific Hilbert 
# space for quantum computing simulations. The `compiler` function integrates 
# various operators acting on different qubits and applies them in a unified 
# Hilbert space. The module also includes functions to create initial wave 
# functions for quantum state initialization.
#
# The module includes the following functions:
# - compiler(ops, locs, n): Compiles operators into a specific Hilbert space.
# - _wrap_identity(data_list, num_bit_list): Helper function to apply identity 
#   operators to the system's Hilbert space.
# - initial_wf(num_bit): Generates the initial wave function |00...0>.
#
# Refs:
# [1] https://github.com/GiggleLiu/QuantumCircuitBornMachine/tree/master
# [2] https://arxiv.org/abs/1804.04168
#
# © Leonardo Lavagna 2024
# @ NESYA https://github.com/NesyaLab
#------------------------------------------------------------------------------

import numpy as np
import scipy.sparse as sps


def compiler(ops, locs, n):
    '''
    Compile operators into a specific Hilbert space.

    Args:
        ops (list or tuple): A list or tuple of operators to be applied to the system.
        locs (list or tuple): The qubit locations on which the operators will act.
        n (int): The total number of qubits in the quantum system.

    Returns:
        scipy.sparse.csr_matrix: The resulting sparse matrix after applying all operators.

    Remark:
        The function compiles the operators into a combined sparse matrix by applying
        each operator to the corresponding qubits, then applying identity operations 
        on the other qubits to form the full Hilbert space.
    '''
    if np.ndim(locs) == 0:  
        locs = [locs]
    if not isinstance(ops, (list, tuple)):  
        ops = [ops]
    locs = np.asarray(locs)  
    #Invert the qubit locations for correct positioning
    locs = n - locs  
    #Sort the qubit locations
    order = np.argsort(locs)  
    #Add boundary conditions
    locs = np.concatenate([[0], locs[order], [n + 1]]) 
    return _wrap_identity([ops[i] for i in order], np.diff(locs) - 1)  


def _wrap_identity(data_list, num_bit_list):
    '''
    Helper function to apply identity operators to the Hilbert space.

    Args:
        data_list (list): A list of operators to be applied to the quantum system.
        num_bit_list (list): A list containing the number of qubits on which each operator acts.

    Returns:
        scipy.sparse.csr_matrix: The resulting sparse matrix after applying the operators.
        
    Raises:
        Exception: If the length of num_bit_list is inconsistent with the number of operators.
    '''
    if len(num_bit_list) != len(data_list) + 1:  # Ensure consistency in input lengths
        raise Exception("Mismatch in number of operators and qubit segments.")
    res = sps.eye(2**num_bit_list[0])  # Initialize the result with the identity matrix for the first segment
    for data, nbit in zip(data_list, num_bit_list[1:]):
        res = sps.kron(res, data)  # Apply the operator to the current segment
        res = sps.kron(res, sps.eye(2**nbit, dtype='complex128'))  # Apply identity operators to the other segments
    return res


def initial_wf(num_bit):
    '''
    Generates the initial wave function |00...0> for a quantum system.

    Args:
        num_bit (int): The number of qubits in the system.

    Returns:
        np.ndarray: The initial wave function as a numpy array.
        
    Remark:
        The function returns the state vector |00...0>, which represents the 
        quantum system being initialized to the ground state (all qubits in state |0>).
    '''
    wf = np.zeros(2**num_bit, dtype='complex128') 
    wf[0] = 1.  # Set the first element to 1, representing the |00...0> state
    return wf
