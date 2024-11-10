
# LSTM for international airline passengers problem with regression framing
import numpy
import pandas as pd
import math
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression,SGDRegressor,Perceptron,SGDClassifier
from sklearn.ensemble import GradientBoostingRegressor
import os
import pickle
import argparse
import joblib

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

    TRAIN_PATH = 'data/ml/' + TEST_NAME +'/training/'
    TEST_PATH = 'data/ml/' + TEST_NAME +'/test/'
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
    
    
    
    ###################################################
    #### choice part of dataset
    ###################################################
    np.random.seed(42)
    train = train[np.random.choice(len(train), int(0.4 * len(train)), replace=False)]
    test = test[np.random.choice(len(test), int(0.4 * len(test)), replace=False)]
    
    validation = load_dataset(VALIDATION_PATH)
    validation = scaler.fit_transform(validation)
    
    
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
        model_ml = SGDRegressor(learning_rate='constant', eta0=0.01, max_iter=1, warm_start=True)
        # model_ml = RandomForestRegressor()
        model_ml.fit(trainX_reshaped, trainY)
        print("*** Model fitted ***")
        print("Saved model to disk")
        # 保存训练好的模型
        with open(MODEL_SAVE_PATH, 'wb') as file:
            pickle.dump(model_ml, file)
        print(f"Model saved to {MODEL_SAVE_PATH}")
        
    
    with open(MODEL_SAVE_PATH, 'rb') as file:
        model = pickle.load(file)
    # make predictions
    trainX_reshaped = trainX.reshape(trainX.shape[0], -1)  
    testX_reshaped = testX.reshape(testX.shape[0], -1)
    validationX_reshaped = validationX.reshape(validationX.shape[0], -1)

    trainPredict = model.predict(trainX_reshaped)
    testPredict = model.predict(testX_reshaped)
    validationPredict = model.predict(validationX_reshaped)
    
    trainScore = r2_score(trainY.flatten(), trainPredict.flatten())
    print('Train Score: %.2f R2' % (trainScore))
    output_file.write('Train Score: %.2f R2\n' % (trainScore))
    testScore = r2_score(testY.flatten(), testPredict.flatten())
    print('Test Score: %.2f R2' % (testScore))
    output_file.write('Test Score: %.2f R2\n' % (testScore))
    validationScore = r2_score(validationY.flatten(), validationPredict.flatten())
    print('Validation Score: %.2f R2' % (validationScore))
    output_file.write('Validation Score: %.2f R2\n' % (validationScore))

    # show_plots.ml_plots(LOG_FILE, TEST_NAME)

if __name__ == "__main__":    

    parser = argparse.ArgumentParser()
    parser.add_argument('-train', action="store_true", default=False)   
    args = parser.parse_args()
    
    RESULTS_PATH = 'results/rf'
    output_file = open(os.path.join(RESULTS_PATH, 'results.txt'), 'w+')
    
    test_names = ["KMeans", "PageRank", "SGD"]
    print("Running all experiments:\n")
    #### tensorflow requires too much time
    print("***********Running models with context***********")
    output_file.write('RUNNING MODELS WITH CONTEXT\n\n')
    for test_name in test_names:
        print("Dataset used: %s" %(test_name))
        output_file.write('CASE: '+ test_name + '\n')
        main(test_name, output_file, context=True, model_train = args.train)
    
    # print("***********Running models without context***********")
    # output_file.write('\n\nRUNNING MODELS WITHOUT CONTEXT\n\n')
    # for test_name in test_names:
    #     print("Dataset used: %s" %(test_name))
    #     output_file.write('CASE: '+ test_name + '\n')
    #     main(test_name, output_file, context=False, model_train = args.train)
    # output_file.close()
