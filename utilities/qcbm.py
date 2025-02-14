from compiler import compiler
from quantum_gates import rot, CNOT


class QCBM:
    """
    Quantum Circuit Born Machine (QCBM) class.
    This class represents a parameterized quantum circuit used for generative modeling.
    """
    def __init__(self, num_qubits, depth):
        self.num_qubits = num_qubits
        self.depth = depth
        self.circuit = self.initialize_circuit()

    def initialize_circuit(self):
        rotations = [rot(0, 0, 0)] * self.num_qubits
        return compiler(rotations, list(range(self.num_qubits)), self.num_qubits)


class BlockQueue:
    """
    Manages a queue of quantum circuit layers.
    """
    def __init__(self, blocks=None):
        self.blocks = blocks if blocks is not None else []

    def add_block(self, block):
        self.blocks.append(block)

    def get_blocks(self):
        return self.blocks


class CNOTEntangler:
    """
    Defines a CNOT-based entanglement layer in the quantum circuit.
    """
    def __init__(self, num_qubits, pairs):
        self.num_qubits = num_qubits
        self.pairs = pairs

    def apply(self):
        return [CNOT(a, b, self.num_qubits) for a, b in self.pairs]


class ArbitraryRotation:
    """
    Defines an arbitrary rotation layer in the quantum circuit.
    """
    def __init__(self, num_qubits):
        self.num_qubits = num_qubits

    def apply(self, angles):
        return [rot(*angles[i]) for i in range(self.num_qubits)]
