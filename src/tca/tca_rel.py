import cupy as cp
import sklearn.metrics
import numpy as np

class TCA:
    def __init__(self, kernel_type='primal', dim=30, lamb=1, gamma=1):
        self.kernel_type = kernel_type
        self.dim = dim
        self.lamb = lamb
        self.gamma = gamma

    def kernel(self, X1, X2, gamma):
        K = None
        ker = self.kernel_type
        if not ker or ker == 'primal':
            K = X1
        elif ker == 'linear':
            if X2 is not None:
                K = sklearn.metrics.pairwise.linear_kernel(
                    np.asarray(X1).T, np.asarray(X2).T)
            else:
                K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1).T)
        elif ker == 'rbf':
            if X2 is not None:
                K = sklearn.metrics.pairwise.rbf_kernel(
                    np.asarray(X1).T, np.asarray(X2).T, gamma)
            else:
                K = sklearn.metrics.pairwise.rbf_kernel(
                    np.asarray(X1).T, None, gamma)
        return K

    def fit(self, Xs, Xt):
        cp.get_default_memory_pool().free_all_blocks()
        Xs = cp.asarray(Xs)
        Xt = cp.asarray(Xt)
        X = cp.hstack((Xs.T, Xt.T))
        X /= cp.linalg.norm(X, axis=0)
        m, n = X.shape
        ns, nt = len(Xs), len(Xt)
        e = cp.vstack((1 / ns * cp.ones((ns, 1)), -1 / nt * cp.ones((nt, 1))))
        M = e @ e.T
        M = M / cp.linalg.norm(M, 'fro')
        H = cp.eye(n) - 1 / n * cp.ones((n, n))
        print("reach here")
        
        K = self.kernel(cp.asnumpy(X), None, 1)  # Convert CuPy array to NumPy array
        K = cp.asarray(K)  # Convert back to CuPy array
        print("finish kernel")
        
        # Release unused memory
        cp.get_default_memory_pool().free_all_blocks()
        n_eye = m if self.kernel_type == 'primal' else n
        a, b = K @ M @ K.T + self.lamb * cp.eye(n_eye), K @ H @ K.T
        print("ready eigen value")
        
        A_inv_B = cp.linalg.solve(a, b)
        
        # Release unused memory
        cp.get_default_memory_pool().free_all_blocks()
        
        # Convert A_inv_B to NumPy array and perform SVD on CPU
        A_inv_B_cpu = cp.asnumpy(A_inv_B)
        U, S, Vt = np.linalg.svd(A_inv_B_cpu)
        print("solve eigen value")
        
        # Convert the results back to CuPy arrays
        U = cp.asarray(U)
        
        A = U[:, :self.dim]
        Z = A.T @ K
        Z /= cp.linalg.norm(Z, axis=0)

        # Release unused memory
        cp.get_default_memory_pool().free_all_blocks()

        Xs_new, Xt_new = Z[:, :ns].T, Z[:, ns:].T
        print("done")
        return cp.asnumpy(Xs_new), cp.asnumpy(Xt_new)