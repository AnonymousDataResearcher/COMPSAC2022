import warnings

import numpy.matlib as npmat
import numpy as np
from numba import njit, b1, f4, f8, u8

"""
Taken from https://github.com/sepehr3pehr/AQBC/blob/master/quantizer.py
Code is cloned, with three small modifications that should not alter the effect:
* The random generation can be seeded with a given seed, to guarantee repeatability
* The initial filling of B is adapted to guarantee that it does not contain zero-vectors. While theoretically possible
    initially, the possibility is (1/2)^nbits. If nbits is low and X.shape[1] is high this can occur. The modification
    guarantees that none of the B columns are all-zero, by setting the bits at random first, and selecting a random bit
    to be high in each column. This means the expected number of high bits is slightly higher than 1/2 * nbits per
    column, but for the effective execution this should be different
* optimize_all no longer prints progress
"""


@njit([f'b1[:,:]({x}[:,::1],i8,{x}[:,::1])' for x in ['f4', 'f8']])
def _jitted_hash(X, n_bits, R):
    B_out = np.empty((n_bits, X.shape[1]), dtype=b1)
    RX = R.T.dot(X)
    for i in range(X.shape[1]):
        args_sort = np.argsort(RX[:, i].T)
        args_sort = args_sort[::-1]
        best_psi = -1 * np.inf
        for k in range(n_bits):
            if RX[args_sort[k], i] == 0:
                break
            b_i = np.zeros(n_bits, dtype=RX.dtype)
            b_i[args_sort[:k + 1]] = 1
            psi = (b_i / np.linalg.norm(b_i)).T.dot(RX[:, i])
            if psi > best_psi:
                best_b_i = b_i
                best_psi = psi

        B_out[:, i] = best_b_i
    return B_out


@njit([f'b1[:,:]({x}[:,::1],i8,{x}[:,::1])' for x in ['f4', 'f8']])
def _jitted_b(X, n_bits, R):
    B = np.empty((n_bits, X.shape[1]), dtype=b1)
    RX = R.T.dot(X)
    for i in range(X.shape[1]):
        args_sort = np.argsort(RX[:, i].T)
        args_sort = args_sort[::-1]
        best_psi = -1 * np.inf
        for k in range(n_bits):
            if RX[args_sort[k], i] == 0:
                break
            b_i = np.zeros(n_bits)
            b_i[args_sort[:k + 1]] = 1
            psi = np.sum(RX[args_sort[:k + 1], i]) / np.sqrt(k + 1)
            if psi > best_psi:
                best_b_i = b_i
                best_psi = psi

        B[:, i] = best_b_i
    return B


@njit([f'{x}[:,:]({x}[:],u8)' for x in ['f4', 'f8']])
def _tile_implementation(arr, reps):
    n = arr.shape[0]
    ret = np.empty((n * reps,), arr.dtype)
    for i in range(reps):
        start = i * n
        end = start + n
        ret[start:end] = arr

    return ret.reshape(reps, -1)


# wrt the original implementation, tile is actually not needed because of the '/' operations shape casting
@njit([f'f8[:, :]({x}[:, :], u8, b1[:, :])' for x in ['f8']])
def _jitted_r(X, n_bits, B):
    normB = np.sqrt(np.sum(B, axis=0))
    BNormalized = B / normB
    U, _, V = np.linalg.svd(X.dot(BNormalized.T))
    return (V.T.dot(U[:, :n_bits].T)).T


class AQBC:
    def __init__(self, X, nbits, epochs, seed=None):
        self.X = X
        self.n = self.X.shape[1]
        self.d = X.shape[0]
        self.nbits = nbits
        self.epochs = epochs

        # Edited w.r.t. original implementation to guarantee same result
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        else:
            self.rng = np.random

        random_b = self.rng.randint(2, size=(self.nbits, self.n))
        set_b = self.rng.rand(self.nbits, self.n)
        set_b = (set_b == np.min(set_b, axis=0))
        self.B = np.maximum(random_b, set_b).astype(np.bool)
        self.R = None

        self.curr_obj = 0

    # R is d * c

    def objective(self):
        normB = np.linalg.norm(self.B, axis=0)
        repNormB = npmat.repmat(normB, self.nbits, 1)
        BNormalized = np.divide(self.B, repNormB)
        RX = self.R.T.dot(self.X)
        self.obj = 0
        for i in range(self.n):
            self.obj += BNormalized[:, i].T.dot(RX[:, i])

    def optimize_all(self):
        for i in range(self.epochs):
            #            U, V = _jitted_r(self.X, self.nbits, self.B)
            self.R = np.ascontiguousarray(_jitted_r(self.X, self.nbits, self.B))
            self.B = _jitted_b(self.X, self.nbits, self.R)

    def hash(self, X):
        return _jitted_hash(X, self.nbits, self.R)

# --- End of cloned code -----------------------------------------------------------------------------------------------
