#------------------------------------------------------------------------------
# qcbm.py
#
# This module provides various quantum circuit components for constructing and 
# simulating quantum circuits used in the Quantum Circuit Born Machine (QCBM) framework.
# The components include arbitrary rotation gates, CNOT entanglers, and the 
# ability to compute probability distributions and loss functions using MMD 
# (Maximum Mean Discrepancy). Additionally, the module includes functionality 
# for numerical gradient computation and sampling from probability distributions.
#
# The module includes the following classes and functions:
# - ArbitraryRotation: Implements an arbitrary rotation gate for quantum circuits.
# - CNOTEntangler: Implements a CNOT entangling layer for quantum circuits.
# - BlockQueue: Keeps track of the quantum circuit's evolution by managing blocks 
#   of operations and applying them to a quantum register.
# - QCBM: Implements the Quantum Circuit Born Machine framework, which uses quantum 
#   circuits to model probability distributions and optimize via MMD loss.
# - Utility functions: Functions for gradient computation, sampling, and probability 
#   distribution manipulation such as `get_nn_pairs`, `sample_from_prob`, 
#   and `prob_from_sample`.
#
# Refs:
# [1] https://github.com/GiggleLiu/QuantumCircuitBornMachine/tree/master
# [2] https://arxiv.org/abs/1804.04168
#
# © Leonardo Lavagna 2024
# @ NESYA https://github.com/NesyaLab
#------------------------------------------------------------------------------

import numpy as np
from utilities.quantum_gates import rot, CNOT
from utilities.compiler import compiler, initial_wf, _rot_tocsr_update1


class ArbitraryRotation(object):
    '''
    Arbitrary rotation gate.

    This class represents a quantum gate that applies arbitrary rotations 
    to qubits in the quantum circuit. It can apply three rotations per qubit, 
    represented by a list of rotation angles.
    '''
    def __init__(self, num_bit):
        '''
        Initializes an ArbitraryRotation instance.
        
        Args:
            num_bit (int): The number of qubits in the quantum system.
        '''
        self.num_bit = num_bit
        self.mask = np.array([True] * (3 * num_bit), dtype='bool')  

    @property
    def num_param(self):
        '''Number of parameters for the rotations (3 parameters per qubit).'''
        return self.mask.sum()

    def tocsr(self, theta_list):
        '''Transforms this block into a sequence of sparse CSR matrices.'''
        theta_list_ = np.zeros(3 * self.num_bit) 
        theta_list_[self.mask] = theta_list  
        rots = [rot(*ths) for ths in theta_list_.reshape([self.num_bit, 3])]  
        res = [compiler([r], [i], self.num_bit) for i, r in enumerate(rots)]  
        return res


class CNOTEntangler(object):
    '''
    CNOT Entangler Layer.

    This class applies a series of CNOT gates that entangle pairs of qubits. 
    The entanglement is formed by applying a CNOT gate for each pair in the provided list.
    '''
    def __init__(self, num_bit, pairs):
        '''
        Initializes a CNOTEntangler instance.
        
        Args:
            num_bit (int): The number of qubits in the quantum system.
            pairs (list of tuples): List of pairs of qubits to apply CNOT gates to.
        '''
        self.num_bit = num_bit
        self.pairs = pairs

    @property
    def num_param(self):
        '''Number of parameters (CNOT gates do not have parameters).'''
        return 0

    def tocsr(self, theta_list):
        '''Transforms this block into a sequence of sparse CSR matrices.'''
        i, j = self.pairs[0]  
        res = CNOT(i, j, self.num_bit)  
        for i, j in self.pairs[1:]:
            res = CNOT(i, j, self.num_bit).dot(res)
        res.eliminate_zeros() 
        return [res]


class BlockQueue(list):
    '''
    Block Queue that keeps track of the theta_list changing history, for fast updates.

    This class stores a sequence of quantum operations (blocks) and efficiently 
    applies them to a quantum register while keeping track of the parameter changes.
    '''
    def __init__(self, *args):
        '''
        Initializes a BlockQueue instance.

        Args:
            *args: A sequence of blocks (quantum operations).
        '''
        list.__init__(self, *args)
        self.theta_last = None
        self.memo = None

    def __call__(self, qureg, theta_list):
        '''
        Apply operations on the quantum register in place.
        
        Args:
            qureg (numpy.ndarray): The quantum register representing the quantum state.
            theta_list (numpy.ndarray): The list of parameters for the quantum gates.
        '''
        remember = self.theta_last is None or (abs(self.theta_last - theta_list) > 1e-12).sum() > 1
        mats = []  
        theta_last = self.theta_last
        if remember:
            self.theta_last = theta_list.copy()  
        qureg_ = qureg 
        for iblock, block in enumerate(self):
            num_param = block.num_param
            theta_i, theta_list = np.split(theta_list, [num_param])
            if theta_last is not None:
                theta_o, theta_last = np.split(theta_last, [num_param])  
            if self.memo is not None and (num_param == 0 or np.abs(theta_i - theta_o).max() < 1e-12):
                mat = self.memo[iblock]  
            else:
                if self.memo is not None and not remember:
                    mat = _rot_tocsr_update1(block, self.memo[iblock], theta_o, theta_i)  
                else:
                    mat = block.tocsr(theta_i)  
            for mat_i in mat:
                qureg_ = mat_i.dot(qureg_) 
            mats.append(mat) 
        if remember:
            self.memo = mats  
        qureg[...] = qureg_  
        np.testing.assert_(len(theta_list) == 0) 

    @property
    def num_bit(self):
        '''Number of qubits in the quantum circuit.'''
        return self[0].num_bit

    @property
    def num_param(self):
        '''Total number of parameters across all blocks.'''
        return sum([b.num_param for b in self])


class QCBM(object):
    '''
    Quantum Circuit Born Machine framework.

    The QCBM is a quantum machine learning model that learns to approximate 
    probability distributions. The circuit consists of rotation gates and CNOT 
    entanglers, and the model is trained using the MMD loss function.
    '''
    def __init__(self, circuit, mmd, p_data, batch_size=None):
        '''
        Initializes a QCBM instance.

        Args:
            circuit (BlockQueue): The sequence of quantum gates (blocks) for the quantum circuit.
            mmd (object): The MMD (Maximum Mean Discrepancy) metric for comparing distributions.
            p_data (numpy.ndarray): The target probability distribution for training.
            batch_size (int, optional): The batch size for sampling.
        '''
        self.circuit = circuit
        self.mmd = mmd
        self.p_data = p_data
        self.batch_size = batch_size

    @property
    def depth(self):
        '''Returns the depth of the circuit, defined by the number of entanglers.'''
        return (len(self.circuit) - 1) // 2

    def pdf(self, theta_list):
        '''
        Get the probability distribution function by applying the circuit.

        Args:
            theta_list (numpy.ndarray): The list of parameters (angles) for the quantum gates.

        Returns:
            numpy.ndarray: The probability distribution function computed from the quantum state.
        '''
        wf = initial_wf(self.circuit.num_bit)  
        self.circuit(wf, theta_list) 
        pl = np.abs(wf)**2  
        # Introduce sampling error if batch_size is provided
        if self.batch_size is not None:
            pl = prob_from_sample(sample_from_prob(np.arange(len(pl)), pl, self.batch_size), len(pl))
        return pl

    def mmd_loss(self, theta_list):
        '''Compute the MMD loss for the given parameters.'''
        self._prob = self.pdf(theta_list)
        return self.mmd(self._prob, self.p_data)

    def gradient(self, theta_list):
        '''
        Compute the gradient using the MMD loss.
        
        Args:
            theta_list (numpy.ndarray): The list of parameters (angles).

        Returns:
            numpy.ndarray: The gradient of the MMD loss with respect to the parameters.
        '''
        prob = self.pdf(theta_list)
        grad = []
        for i in range(len(theta_list)):
            # Numerical gradient computation
            theta_list[i] += np.pi / 2.
            prob_pos = self.pdf(theta_list)
            theta_list
