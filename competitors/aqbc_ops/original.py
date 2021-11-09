import numpy.matlib as npmat
import numpy as np
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
        self.B = np.maximum(random_b, set_b)

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

    def optimize_B(self):
        B = np.zeros((self.nbits, self.n))
        RX = self.R.T.dot(self.X)
        for i in range(self.n):
            args_sort = np.argsort(RX[:, i].T)
            args_sort = args_sort[::-1]
            best_psi = -1 * np.inf
            for k in range(self.nbits):
                if (RX[args_sort[k], i] == 0):
                    break
                b_i = np.zeros((1, self.nbits))
                b_i[:, args_sort[:k + 1]] = 1
                psi = np.sum(RX[args_sort[:k + 1], i]) / np.sqrt(k + 1)
                if (psi > best_psi):
                    best_b_i = b_i
                    best_psi = psi

            self.B[:, i] = best_b_i

    def optimize_R(self):
        normB = np.linalg.norm(self.B, axis=0)
        repNormB = np.matlib.repmat(normB, self.nbits, 1)
        BNormalized = np.divide(self.B, repNormB)
        U, _, V = np.linalg.svd(self.X.dot(BNormalized.T))
        self.R = U[:, :self.nbits].dot(V)

    def optimize_all(self):
        for i in range(self.epochs):
            # print("iteration {}".format(i))
            self.optimize_R()
            self.optimize_B()
            if i % 2 == 0:
                self.objective()
                # print("obj @ {} is {}".format(i, self.obj))

    def hash(self, X):
        B_out = self.rng.randint(2, size=(self.nbits, X.shape[1]))
        RX = self.R.T.dot(X)
        for i in range(X.shape[1]):
            args_sort = np.argsort(RX[:, i].T)
            args_sort = args_sort[::-1]
            best_psi = -1 * np.inf
            for k in range(self.nbits):
                if (RX[args_sort[k], i] == 0):
                    break
                b_i = np.zeros((1, self.nbits))
                b_i[:, args_sort[:k + 1]] = 1
                psi = RX[:, i].T.dot(np.squeeze(b_i)
                                     / np.linalg.norm(b_i))
                if (psi > best_psi):
                    best_b_i = b_i
                    best_psi = psi

            B_out[:, i] = best_b_i
        return B_out


# --- End of cloned code -----------------------------------------------------------------------------------------------