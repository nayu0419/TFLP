import numpy as np
import tensorly.tenalg as tl
import scipy as sp


class Model(object):
    def __init__(self, name='trpca_tnn'):
        super().__init__()
        self.name = name

    def tubalrank(self, X, tol):
        X = np.fft.fft(X, axis=2)
        n1, n2, n3 = X.shape
        s = np.zeros((np.min([n1, n2]), 1))

        # i=0
        s = s + np.linalg.svd(X[:, :, 0], full_matrices=False)

        # i=1,...,halfn3
        halfn3 = np.round(n3 / 2)
        for i in range(1, halfn3):
            s = s + np.linalg.svd(X[:, :, i], full_matrices=False) * 2

        # if n3 is even
        if np.mod(n3, 2) == 0:
            i = halfn3
            s = s + np.linalg.svd(X[:, :, i], full_matrices=False)
        s = s / n3

        # Check for this line
        # if nargin==1
        # tol = np.max([n1,n2]) * eps(np.max(s));
        trank = np.sum(s[s > tol])

        return trank

    def tprod(self, A, B):
        n1, n2, n3 = A.shape
        m1, m2, m3 = B.shape

        if n2 != m1 or n3 != m3:
            raise ValueError("Inner tensor dimensions must agree.")

        Af = np.fft.fft(A, axis=2)
        Bf = np.fft.fft(B, axis=2)
        Cf = np.zeros((n1, m2, n3), dtype=complex)

        # first frontal slice
        Cf[:, :, 0] = Af[:, :, 0].dot(Bf[:, :, 0])

        # i=2,...,halfn3
        halfn3 = int(np.round(n3 / 2))
        # print("halfn3: ", halfn3)
        for i in range(1, halfn3):
            Cf[:, :, i] = Af[:, :, i].dot(Bf[:, :, i])
            Cf[:, :, n3 - i] = np.conj(Cf[:, :, i])  # CHECK INDEXING
        # print("i: ", i, ", n3-i: ", n3-i)

        # if n3 is even
        if np.mod(n3, 2) == 0:
            i = halfn3
            # print("Even: ", i)
            Cf[:, :, i] = Af[:, :, i].dot(Bf[:, :, i])

        C = np.fft.ifft(Cf, axis=2)

        return C, Af, Bf, Cf


    def prox_tnn(self, Y, rho):
        n1, n2, n3 = Y.shape
        X = np.zeros(Y.shape, dtype=complex)
        Y = np.fft.fft(Y, axis=2)
        tnn = 0
        trank = 0

        # first frontal slice
        U, S, V = np.linalg.svd(Y[:, :, 0], full_matrices=False)
        r = len(S[S > rho])
        if r >= 1:
            S = S[0:r] - rho
            X[:, :, 0] = U[:, 0:r].dot(np.diag(S)).dot(V.T[:, 0:r].T)
            tnn = tnn + np.sum(S)
            trank = np.max([trank, r])

        # i=2,...,halfn3
        halfn3 = int(np.round(n3 / 2))
        for i in range(1, halfn3):
            U, S, V = np.linalg.svd(Y[:, :, i], full_matrices=False)
            r = len(S[S > rho])
            if r >= 1:
                S = S[0:r] - rho
                X[:, :, i] = U[:, 0:r].dot(np.diag(S)).dot(V.T[:, 0:r].T)
                tnn = tnn + np.sum(S) * 2
                trank = np.max([trank, r])
            X[:, :, n3 - i] = np.conj(X[:, :, i])

        # if n3 is even
        if np.mod(n3, 2) == 0:
            U, S, V = np.linalg.svd(Y[:, :, halfn3], full_matrices=False)
            r = len(S[S > rho])
            if r >= 1:
                S = S[0:r] - rho
                X[:, :, halfn3] = U[:, 0:r].dot(np.diag(S)).dot(V.T[:, 0:r].T)
                tnn = tnn + np.sum(S)
                trank = np.max([trank, r])

        # Output results
        tnn = tnn / n3
        X = np.fft.ifft(X, axis=2)

        return X, tnn

    #
    def prox_l1(self, b, lambda_):
        return np.maximum(0, b - lambda_) + np.minimum(0, b + lambda_)

    # Solve the Tensor Robust Principal Component Analysis (TRPCA) based on
    # Tensor Nuclear Norm (TNN) problem by ADMM:
    #
    # min_{L,S} ||L||_*+lambda*||S||_1, s.t. X=L+S
    #
    # ---------------------------------------------
    # Input:
    #       X       -    d1*d2*d3 tensor
    #       lambda  -    > 0, parameter
    #       opts    -    Structure value in Matlab. The fields are
    #           opts.tol        -   termination tolerance
    #           opts.max_iter   -   maximum number of iterations
    #           opts.mu         -   stepsize for dual variable updating in ADMM
    #           opts.max_mu     -   maximum stepsize
    #           opts.rho        -   rho>=1, ratio used to increase mu
    #           opts.DEBUG      -   0 or 1
    #
    # Output:
    #       L       -    d1*d2*d3 tensor
    #       S       -    d1*d2*d3 tensor
    #       obj     -    objective function value
    #       err     -    residual
    #       iter    -    number of iterations
    #
    # version 1.0 - 19/06/2016
    #
    # Written by Canyi Lu (canyilu@gmail.com)
    # Ported by Fernando Hermosillo

    #

    def trpca_tnn(self, X, S_d, S_m, TT, lambda_=713, mu=1e-2, alpha=0.3):
        # Options structure
        # Options = namedtuple("Options", "tol max_iter rho mu max_mu DEBUG")
        # result = namedtuple('Result',result._fields+('point',))

        # Default options
        tol = 1e-5
        max_iter = 500  # 500
        rho = 1.1
        max_mu = 1e10
        DEBUG = False

        # Lambda
        dim = X.shape

        # Initialize L, S and Y
        L = np.zeros((dim))
        S = L
        Y = L

        ## ITERATIVE PROCESS
        for itercount in range(0, max_iter):
            Lk = L
            Sk = S

            # update L
            L, tnnL = self.prox_tnn(-S + X - Y / mu, 1 / mu)

            # update S
            S = self.prox_l1(-L + X - Y / mu, lambda_ / mu)

            # Compute residual error
            dY = L + S - X
            chgL = np.max(np.abs(Lk.flatten() - L.flatten()))
            chgS = np.max(np.abs(Sk.flatten() - S.flatten()))
            chg = np.max([chgL, chgS, np.max(np.abs(dY.flatten()))])
            # Debug
            if DEBUG:
                if itercount == 1 or np.mod(itercount, 10) == 0:
                    obj = tnnL + lambda_ * np.linalg.norm(S[:], ord=1)
                    err = np.linalg.norm(dY[:])
                    print("iter ", iter, ", mu=", mu, ", obj=", obj, ", err=", err)

            # Stop condition
            if chg < tol:
                break
            Y = Y + mu * dY
            mu = np.min([rho * mu, max_mu])

        #obj = tnnL + lambda_ * np.linalg.norm(S.flatten().flatten(), ord=1)
        #err = np.linalg.norm(dY.flatten().flatten())

        L1 = L.real
        L2 = L.real

        delta = 1
        while(delta > 1e-6):
            l1 = alpha * tl.mode_dot(L1, S_d, 1) + (1 - alpha) * TT
            cha = l1 - L1
            delta = np.linalg.norm(cha.flatten().flatten(), ord=1)
            L1 = l1
        delta = 1
        while (delta > 1e-6):
            l2 = alpha * tl.mode_dot(L2, S_m, 0) + (1 - alpha) * TT
            cha = l2 - L2
            delta = np.linalg.norm(cha.flatten().flatten(), ord=1)
            L2 = l2
        predict_X = (l1 + l2)/2
        return predict_X


    def __call__(self):

        return getattr(self, self.name, None)
