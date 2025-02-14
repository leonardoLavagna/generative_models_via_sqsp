#------------------------------------------------------------------------------
# quantum_gates.py
#
# This module contains functions and operators commonly used in quantum circuits 
# for simulation and manipulation of qubit states. The module defines various 
# gates such as single qubit rotations, a general rotation gate, and the CNOT gate. 
# These functions are crucial for building and simulating quantum circuits for 
# quantum computing applications.
#
# The module includes the following functions:
# - _ri(si, theta): Generates a single qubit rotation operator for a given angle.
# - rot(t1, t2, t3): Generates a general rotation gate consisting of rz(t3)rx(t2)rz(t1).
# - CNOT(ibit, jbit, n): Generates a CNOT gate for specified qubit positions.
#
# Refs:
# [1] https://github.com/GiggleLiu/QuantumCircuitBornMachine/tree/master
# [2] https://arxiv.org/abs/1804.04168
#
# © Leonardo Lavagna 2025
# @ NESYA https://github.com/NesyaLab
#------------------------------------------------------------------------------

import numpy as np
import scipy.sparse as sps
from compiler import compiler


# Define basic operators
I2 = sps.eye(2).tocsr()  # Identity matrix for 1 qubit
sx = sps.csr_matrix([[0, 1], [1, 0.]])  # Pauli-X matrix (X gate)
sy = sps.csr_matrix([[0, -1j], [1j, 0.]])  # Pauli-Y matrix (Y gate)
sz = sps.csr_matrix([[1, 0], [0, -1.]])  # Pauli-Z matrix (Z gate)

# Projection operators for qubits in state |0> and |1>
p0 = (sz + I2) / 2
p1 = (-sz + I2) / 2


def _ri(si, theta):
    """
    Generates a single qubit rotation operator for a given angle.

    Args:
        si (scipy.sparse.csr_matrix): Pauli matrix (X, Y, or Z).
        theta (float): Rotation angle.

    Returns:
        scipy.sparse.csr_matrix: The single qubit rotation matrix.
        
    Remarks:
        The function applies the rotation based on the formula:
        cos(theta/2) * I - i*sin(theta/2) * si, where I is the identity matrix.
    """
    return np.cos(theta / 2.) * I2 - 1j * np.sin(theta / 2.) * si


def rot(t1, t2, t3):
    """
    Generates a general rotation gate rz(t3)rx(t2)rz(t1).

    Args:
        t1 (float): Angle for the first rz rotation.
        t2 (float): Angle for the rx rotation.
        t3 (float): Angle for the second rz rotation.

    Returns:
        scipy.sparse.csr_matrix: The combined rotation gate.
        
    Remarks:
        The function applies a sequence of rotations using the formula:
        Rz(t3) Rx(t2) Rz(t1), where Rz is a rotation about the z-axis 
        and Rx is a rotation about the x-axis.
    """
    return _ri(sz, t3).dot(_ri(sx, t2)).dot(_ri(sz, t1))


def CNOT(ibit, jbit, n):
    """
    Generates a CNOT (Controlled-NOT) gate for the specified qubit positions.

    Args:
        ibit (int): The control qubit position.
        jbit (int): The target qubit position.
        n (int): Total number of qubits in the system.

    Returns:
        scipy.sparse.csr_matrix: The CNOT gate as a sparse matrix.

    Remarks:
        The function uses the compiler to build the CNOT gate based on projection 
        operators (p0, p1) and Pauli-X operator, applying it to the qubits ibit and jbit.
    """
    res = compiler([p0, I2], [ibit, jbit], n)
    res = res + compiler([p1, sx], [ibit, jbit], n)
    return res
