
# LSTM for international airline passengers problem with regression framing
import numpy
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Activation
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
import os
import pickle 
import timeit
from show_plots import ml_plots
import argparse

def create_dataset(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), :].copy()
        a[-1,-1] = 0
        dataX.append(a.flatten())
        dataY.append(dataset[i + look_back - 1, -1])
    return numpy.array(dataX), numpy.array(dataY)

def load_dataset(path):
    dfs = []
    for f in os.listdir(path):
        df = pd.read_csv(path + f, engine='python', skipfooter=1)
        df = df.drop(columns=['index'])
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    dataset = df.values
    dataset = dataset.astype('float32')

    dataset = dataset[:-1,:]
    
    return dataset

def main(TEST_NAME, output_file, context, model_train):
    
    numpy.random.seed(7)
    
    TRAIN_PATH = 'data/ml/' + TEST_NAME +'/training/'
    TEST_PATH = 'data/ml/' + TEST_NAME +'/test/'
    VALIDATION_PATH = 'data/ml/' + TEST_NAME +'/validation/'
    
    if context==True:
        MODEL_SAVE_PATH = 'model/ffnn/with_context/model' + TEST_NAME + '.h5'
        LOG_FILE = 'results/ffnn/with_context/model' + TEST_NAME + '.pkl'
    elif context==False:
        MODEL_SAVE_PATH = 'model/ffnn/without_context/model' + TEST_NAME + '.h5'
        LOG_FILE = 'results/ffnn/without_context/model' + TEST_NAME + '.pkl'
    else:
        pass
    
    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    train = load_dataset(TRAIN_PATH)
    print("*** Training dataset loaded ***")
    train = scaler.fit_transform(train)
    
    test = load_dataset(TEST_PATH)
    print("*** Test dataset loaded ***")
    test = scaler.transform(test)
    
    validation = load_dataset(VALIDATION_PATH)
    print("*** Validation dataset loaded ***")
    validation = scaler.transform(validation)
    
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
    
    if model_train==True:
        model = Sequential()
        model.add(Dense(5, input_dim=trainX.shape[1], activation='relu'))
        model.add(Dense(5, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='mean_absolute_error', optimizer='adam')
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10, restore_best_weights=True)

        history = model.fit(trainX, trainY,
                        validation_data=(validationX, validationY),
                        epochs=200, batch_size=20, verbose=1, callbacks=[es])
    
        model.save(MODEL_SAVE_PATH)
        print("*** Model fitted ***")
        print("Saved model to disk")
    
        with open(LOG_FILE, 'wb') as f:
            pickle.dump(model.history.history, f)
    
    # make predictions
    model = load_model(MODEL_SAVE_PATH)
    
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)
    validationPredict = model.predict(validationX)
    
    trainScore = r2_score(trainY.flatten(), trainPredict.flatten())
    print('Train Score: %.2f R2' % (trainScore))
    output_file.write('Train Score: %.2f R2\n' % (trainScore))
    testScore = r2_score(testY.flatten(), testPredict.flatten())
    print('Test Score: %.2f R2' % (testScore))
    output_file.write('Test Score: %.2f R2\n' % (testScore))
    validationScore = r2_score(validationY.flatten(), validationPredict.flatten())
    print('Validation Score: %.2f R2' % (validationScore))
    output_file.write('Validation Score: %.2f R2\n' % (validationScore))
    
    ml_plots(LOG_FILE, TEST_NAME)

if __name__ == "__main__":  
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-train', action="store_true", default=False)   
    args = parser.parse_args()
    
    RESULTS_PATH = 'results/ffnn'
    output_file = open(os.path.join(RESULTS_PATH, 'results.txt'), 'w+')
    test_names = ["KMeans", "PageRank", "SGD"]
    print("Running all experiments:\n")
    #### tensorflow requires too much time
    print("***********Running models with context***********")
    output_file.write('RUNNING MODELS WITH CONTEXT\n\n')
    for test_name in test_names:
        print("Dataset used: %s" %(test_name))
        output_file.write('CASE: '+ test_name + '\n')
        main(test_name, output_file, context=True, model_train=args.train)
    
    print("***********Running models without context***********")
    output_file.write('\n\nRUNNING MODELS WITHOUT CONTEXT\n\n')
    for test_name in test_names:
        print("Dataset used: %s" %(test_name))
        output_file.write('CASE: '+ test_name + '\n')
        main(test_name, output_file, context=False, model_train=args.train)
        
    output_file.close()
