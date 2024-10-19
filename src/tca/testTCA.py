
# LSTM for international airline passengers problem with regression framing
import numpy
import pandas as pd
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
# from skgarden import RandomForestQuantileRegressor
from sklearn.ensemble import RandomForestRegressor
from quantile_forest import RandomForestQuantileRegressor
import os
import pickle
import argparse
import numpy as np
import joblib
import matplotlib.pyplot as plt

rf_quantiles = [0.0, 1.0]
n_tree = 500
tree_max_depth = 10


def pad_features(datasets):
    # Find the maximum number of features across all datasets
    max_features = max([dataset.shape[1] for dataset in datasets])
    
    padded_datasets = []
    for dataset in datasets:
        if dataset.shape[1] < max_features:
            # Pad with zeros to make the number of features equal to max_features
            padded_dataset = np.pad(dataset, ((0, 0), (0, max_features - dataset.shape[1])), 'constant')
        else:
            padded_dataset = dataset
        padded_datasets.append(padded_dataset)
    
    return padded_datasets

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

# 预测并计算分位数
def predict_quantiles(model, X, quantiles, Y):
    preds = np.array([tree.predict(X) for tree in model.estimators_])
    # predict_bound = np.percentile(preds, [q * 100 for q in quantiles], axis=0)
    bounds = model.predict(X, quantiles=quantiles)
    # 检查真实值是否在边界之间
    is_within_bounds = (Y >= bounds[:,0]) & (Y <= bounds[:,1])
     # 计算 True 的百分比
    return np.mean(is_within_bounds) * 100

def main(TEST_NAME, output_file, context, model_train=False):
    
    numpy.random.seed(7)

    TRAIN_PATH = 'data/ml/' + TEST_NAME +'/training/'
    # TEST_PATH = 'data/ml/' + TEST_NAME +'/test/'
    TEST_PATH = 'data/ml/' + 'PageRank' +'/test/'
    VALIDATION_PATH = 'data/ml/' + TEST_NAME +'/validation/'
    
    if context==True:
        MODEL_SAVE_PATH = 'model/rf/with_context/model' + TEST_NAME + '.h5'
        LOG_FILE = 'results/rf/with_context/model' + TEST_NAME + '.pkl'
    elif context==False:
        MODEL_SAVE_PATH = 'model/rf/without_context/model' + TEST_NAME + '.h5'
        LOG_FILE = 'results/rf/without_context/model' + TEST_NAME + '.pkl'
    else:
        pass
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    train = load_dataset(TRAIN_PATH)
    train = scaler.fit_transform(train)
    
    test = load_dataset(TEST_PATH)
    test = scaler.fit_transform(test)
    
    validation = load_dataset(VALIDATION_PATH)
    validation = scaler.fit_transform(validation)
    
    
    train = train[np.random.choice(len(train), int(0.1 * len(train)), replace=False)]
    test = test[np.random.choice(len(test), int(0.1 * len(test)), replace=False)]
    validation = validation[np.random.choice(len(validation), int(0.1 * len(validation)), replace=False)]
    
    datasets = [train, test]
    train, test = pad_features(datasets)
    
    
    if context==True:
        look_back = 5
        trainX, trainY = create_dataset(train, look_back)
        testX, testY = create_dataset(test, look_back)
        validationX, validationY = create_dataset(validation, look_back)
    elif context==False:
        look_back = 1
        trainX, trainY = create_dataset(train, look_back)
        testX, testY = create_dataset(test, look_back)
        validationX, validationY = create_dataset(validation, look_back)
    else:
        pass
    
    
    trainX = numpy.reshape(trainX, (trainX.shape[0], train.shape[1], trainX.shape[1]))


    testX = numpy.reshape(testX, (testX.shape[0], test.shape[1], testX.shape[1]))
    validationX = numpy.reshape(validationX, (validationX.shape[0], validation.shape[1], validationX.shape[1]))
    
    if model_train==True:
        trainX_reshaped = trainX.reshape(trainX.shape[0], -1)  # 转换为 (63769, 75)
        model = RandomForestQuantileRegressor(n_estimators=n_tree, max_depth=tree_max_depth, random_state=10, n_jobs=30)
        model.fit(trainX_reshaped, trainY)
        print("*** Model fitted ***")
        print("Saved model to disk")
        # 保存训练好的模型
        with open(MODEL_SAVE_PATH, 'wb') as file:
            pickle.dump(model, file)
        print(f"Model saved to {MODEL_SAVE_PATH}")
    
    # 加载模型
    with open(MODEL_SAVE_PATH, 'rb') as file:
        model = pickle.load(file)
    # make predictions
    # 转换为 (63769, 75)
    trainX_reshaped = trainX.reshape(trainX.shape[0], -1)  
    testX_reshaped = testX.reshape(testX.shape[0], -1)
    validationX_reshaped = validationX.reshape(validationX.shape[0], -1)

    train_percentage_true = predict_quantiles(model, trainX_reshaped, rf_quantiles, trainY)
    test_percentage_true = predict_quantiles(model, testX_reshaped, rf_quantiles, testY)
    validation_percentage_true = predict_quantiles(model, validationX_reshaped, rf_quantiles, validationY)
    
    print('Trainning Data Bounded: ' + str(train_percentage_true) + ' %')
    print('Testing Data Bounded: ' + str(test_percentage_true) + ' %')
    print('Vali Data Bounded: ' + str(validation_percentage_true) + ' %')

if __name__ == "__main__":    

    parser = argparse.ArgumentParser()
    parser.add_argument('-train', action="store_true", default=False)   
    args = parser.parse_args()
    
    RESULTS_PATH = 'results/rf'
    output_file = open(os.path.join(RESULTS_PATH, 'results.txt'), 'w+')
    
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
