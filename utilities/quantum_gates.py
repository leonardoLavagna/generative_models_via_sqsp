import numpy as np
import scipy.sparse as sps


I2 = sps.eye(2).tocsr()
sx = sps.csr_matrix([[0,1],[1,0.]])
sy = sps.csr_matrix([[0,-1j],[1j,0.]])
sz = sps.csr_matrix([[1,0],[0,-1.]])

def rot_matrix(si, theta):
    """
    Single qubit rotation
    """
    return np.cos(theta/2.)*I2 - 1j*np.sin(theta/2.)*si


def rot(t1, t2, t3):
    """
    A general rotation gate rz(t3)rx(r2)rz(t1).
    """
    return rot_matrix(sz, t3).dot(rot_matrix(sx, t2)).dot(rot_matrix(sz, t1))


def CNOT(ibit, jbit, n):
    """
    CNOT gate
    """
    res = compiler([p0, I2], [ibit, jbit], n)
    res = res + compiler([p1, sx], [ibit, jbit], n)
    return res
