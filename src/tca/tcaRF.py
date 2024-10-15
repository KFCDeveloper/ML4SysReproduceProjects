
# LSTM for international airline passengers problem with regression framing
import numpy
import pandas as pd
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from skgarden import RandomForestQuantileRegressor
import os
import pickle
import argparse
import joblib
import math
import numpy as np
import sklearn.metrics
from scipy.linalg import eigh
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator
from scipy.sparse.linalg import eigs
from scipy.spatial.distance import cdist as sp_cdist

import cupy as cp
import numpy as np
import sklearn.metrics
from cupy.linalg import inv
from scipy.sparse.linalg import eigs



# class TCA:
#     def __init__(self, kernel_type='linear', dim=30, lamb=1, gamma=1):
#         self.kernel_type = kernel_type
#         self.dim = dim
#         self.lamb = lamb
#         self.gamma = gamma
#         self.scaler = MinMaxScaler()

#     def kernel(self, X1, X2=None):
#         ker = self.kernel_type
#         if ker == 'primal':
#             return X1
#         elif ker == 'linear':
#             return sklearn.metrics.pairwise.linear_kernel(X1, X2) if X2 is not None else sklearn.metrics.pairwise.linear_kernel(X1)
#         elif ker == 'rbf':
#             return sklearn.metrics.pairwise.rbf_kernel(X1, X2, self.gamma) if X2 is not None else sklearn.metrics.pairwise.rbf_kernel(X1, None, self.gamma)
#         elif ker == 'gaussian':
#             if X2 is None:
#                 X2 = X1
#             pairwise_sq_dists = euclidean_distances(X1, X2, squared=True)
#             return np.exp(-pairwise_sq_dists / (2 * (self.gamma ** 2)))
#         else:
#             raise ValueError(f"Unknown kernel type: {self.kernel_type}")

#     def fit(self, Xs, Xt):
#         X = np.vstack((Xs, Xt))
#        # X /= np.linalg.norm(X, axis=1, keepdims=True)
#         ns, nt = len(Xs), len(Xt)

#         # Create the centering matrix H
#         H = np.eye(ns + nt) - np.ones((ns + nt, ns + nt)) / float(ns + nt)

#         # Compute the kernel matrix
#         K = self.kernel(X)  # Kernel computation

#         # Create the coefficient matrix L
#         L = np.vstack((
#             np.hstack((np.ones((ns, ns)) / ns ** 2, -1.0 * np.ones((ns, nt)) / (ns * nt))),
#             np.hstack((
#                 -1.0 * np.ones((nt, ns)) / (ns * nt),
#                 np.ones((nt, nt)) / (nt ** 2)))
#         ))

#         mu = self.lamb
#         I = np.eye(ns + nt)
#         KLK = np.dot(K, np.dot(L, K))
#         KHK = np.dot(K, np.dot(H, K))

#         J_cpu = np.dot(np.linalg.inv(I + mu * KLK), KHK)

#         max_dim = J_cpu.shape[0]
#         if self.dim > max_dim:
#             print(f"Warning: Requested dimension {self.dim} is greater than the available eigenvalues {max_dim}. Reducing dim to {max_dim}.")
#             self.dim = max_dim  # Limit self.dim to the maximum number of available eigenvalues

#         # Eigenvector decomposition as solution to trace minimization
#         _, C = eigh(J_cpu, eigvals=(0, self.dim - 1))

#         # Transformation/embedding matrix
#         C_ = np.real(C)  # Take the real part
#         print("TCA is done")
#         # Transform the source data
#         Xs_new = np.dot(K[:ns, :], C_)
#         Xt_new = np.dot(K[ns:, :], C_)
#         Xs_new = self.scaler.fit_transform(Xs_new)  # Fit and transform the training data
#         Xt_new = self.scaler.transform(Xt_new)

#         return Xs_new, Xt_new

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
        if self.kernel_type == 'gaussian':
            if X2 is None:
                X2 = X1
            pairwise_sq_dists = sklearn.metrics.pairwise.euclidean_distances(X1, X2, squared=True)
            K = np.exp(-pairwise_sq_dists / (2 * (self.gamma ** 2)))
            return cp.asarray(K)
        return K

    def fit(self, Xs, Xt):
        print(self.dim)
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
        
        A = U[:, :int(self.dim)] 

        Z = A.T @ K
        Z /= cp.linalg.norm(Z, axis=0)

        # Release unused memory
        cp.get_default_memory_pool().free_all_blocks()

        Xs_new, Xt_new = Z[:, :ns].T, Z[:, ns:].T
        print("done")
        return cp.asnumpy(Xs_new), cp.asnumpy(Xt_new)


def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), :].copy()
        a[-1,-1] = 0
        dataX.append(a)
        dataY.append(dataset[i + look_back - 1, -1])
    return numpy.array(dataX), numpy.array(dataY)

def load_dataset(path, cut = -1):
    dfs = []
    for f in os.listdir(path):
        df = pd.read_csv(path + f, engine='python', skipfooter=1)
        df = df.drop(columns=['index'])
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    dataset = df.values
    dataset = dataset.astype('float32')

    if cut != -1:
        dataset = dataset[:cut,:]

    return dataset



def main(TEST_NAME, output_file, context, model_train=False):
    
    numpy.random.seed(7)

    TRAIN_PATH = 'data/ml/' + TEST_NAME + '/training/'
    TEST_PATH = 'data/ml/' + 'PageRank' + '/test/'
    VALIDATION_PATH = 'data/ml/' + TEST_NAME + '/validation/'
    
    if context:
        MODEL_SAVE_PATH = 'model/rf/with_context/model' + TEST_NAME + '.h5'
        LOG_FILE = 'results/rf/with_context/model' + TEST_NAME + '.pkl'
    else:
        MODEL_SAVE_PATH = 'model/rf/without_context/model' + TEST_NAME + '.h5'
        LOG_FILE = 'results/rf/without_context/model' + TEST_NAME + '.pkl'
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    train = load_dataset(TRAIN_PATH)
    train = scaler.fit_transform(train)
    
    test = load_dataset(TEST_PATH)
    test = scaler.fit_transform(test)
    
    validation = load_dataset(VALIDATION_PATH)
    validation = scaler.fit_transform(validation)
    
    train, _ = train_test_split(train, test_size=0.9, random_state=7)
    test, _ = train_test_split(test, test_size=0.9, random_state=7)
    validation , _ = train_test_split(validation, test_size=0.9, random_state=7)
    
    look_back = 5 if context else 1
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)
    validationX, validationY = create_dataset(validation, look_back)
    
    # Check for feature dimension mismatch
    if trainX.shape[2] != testX.shape[2]:
        print(f"Warning: Mismatch in feature dimensions. Training has {trainX.shape[2]} features, but test has {testX.shape[2]} features.")
        if trainX.shape[2] > testX.shape[2]:
            diff = trainX.shape[2] - testX.shape[2]
            testX = numpy.pad(testX, ((0, 0), (0, 0), (0, diff)), 'constant')
    
    # Reshape train, test, validation data
    trainX_reshaped = trainX.reshape(trainX.shape[0], -1)
    testX_reshaped = testX.reshape(testX.shape[0], -1)
    
    # print(trainX_reshaped.shape)
    # tca = TCA(kernel_type='primal', dim=trainX_reshaped.shape[1], lamb=0.1, gamma=1)
    # # # Fit TCA with source (training) and target (testing) data
    # _, Xt_new = tca.fit(trainX_reshaped,  testX_reshaped)
    # testX_reshaped = Xt_new
    
    
    validationX_reshaped = validationX.reshape(validationX.shape[0], -1)
    
    # Split testX_reshaped into test_trainX and test_testX
    test_trainX, test_testX, test_trainY, test_testY = train_test_split(testX_reshaped, testY, test_size=0.5, random_state=42)

    if model_train:
        # Train the model on trainX_reshaped
        qrf = RandomForestQuantileRegressor(n_estimators=100, random_state=42)
        qrf.fit(trainX_reshaped, trainY)
        print("*** Model fitted on train data ***")
        
        
        ###############################
        #TODO: TCA
        
        ###############################
        
        # Further train the model on test_trainX
        qrf.fit(test_trainX, test_trainY)
        print("*** Model fitted on test_trainX ***")
        
        # Save the model
        with open(MODEL_SAVE_PATH, 'wb') as file:
            pickle.dump(qrf, file)
        print(f"Model saved to {MODEL_SAVE_PATH}")
    
    # Load the model
    with open(MODEL_SAVE_PATH, 'rb') as file:
        model = pickle.load(file)
    
    # Make predictions
    trainPredict = model.predict(trainX_reshaped)
    test_testPredict = model.predict(test_testX)
    validationPredict = model.predict(validationX_reshaped)
    
    # Calculate R2 scores
    trainScore = r2_score(trainY.flatten(), trainPredict.flatten())
    print(f'Train Score: {trainScore:.2f} R2')
    output_file.write(f'Train Score: {trainScore:.2f} R2\n')
    
    testScore = r2_score(test_testY.flatten(), test_testPredict.flatten())
    print(f'Test Score: {testScore:.2f} R2')
    output_file.write(f'Test Score: {testScore:.2f} R2\n')
    
    validationScore = r2_score(validationY.flatten(), validationPredict.flatten())
    print(f'Validation Score: {validationScore:.2f} R2')
    output_file.write(f'Validation Score: {validationScore:.2f} R2\n')

    # Optionally plot the results
    # show_plots.ml_plots(LOG_FILE, TEST_NAME)



if __name__ == "__main__":    

    parser = argparse.ArgumentParser()
    parser.add_argument('-train', action="store_true", default=False)   
    args = parser.parse_args()
    
    RESULTS_PATH = 'results/rf'
    output_file = open(os.path.join(RESULTS_PATH, 'results.txt'), 'w+')
    
    # test_names = ["KMeans", "PageRank", "SGD"]
    test_names = ["KMeans"]
    print("Running all experiments:\n")
    #### tensorflow requires too much time
    print("***********Running models with context***********")
    output_file.write('RUNNING MODELS WITH CONTEXT\n\n')
    for test_name in test_names:
        print("Dataset used: %s" %(test_name))
        output_file.write('CASE: '+ test_name + '\n')
        main(test_name, output_file, context=True, model_train = args.train)
    
    print("***********Running models without context***********")
    output_file.write('\n\nRUNNING MODELS WITHOUT CONTEXT\n\n')
    for test_name in test_names:
        print("Dataset used: %s" %(test_name))
        output_file.write('CASE: '+ test_name + '\n')
        main(test_name, output_file, context=False, model_train = args.train)
    output_file.close()

