import numpy as np
import scipy.sparse as sps


def compiler(ops, locs, n):
    """
    Compile operators into specific Hilbert space.
    """
    if np.ndim(locs) == 0:
        locs = [locs]
    if not isinstance(ops, (list, tuple)):
        ops = [ops]
    locs = np.asarray(locs)
    locs = n - locs
    order = np.argsort(locs)
    locs = np.concatenate([[0], locs[order], [n + 1]])
    return wrap_identity([ops[i] for i in order], np.diff(locs) - 1)


def wrap_identity(data_list, num_bit_list):
    if len(num_bit_list) != len(data_list) + 1:
        raise Exception()
    res = sps.eye(2**num_bit_list[0])
    for data, nbit in zip(data_list, num_bit_list[1:]):
        res = sps.kron(res, sps.eye(2**nbit))
        res = sps.kron(res, data)
    return res
