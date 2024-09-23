import xgboost
import os
import xgboost_util
import math
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import pickle
import logging
import matplotlib.pyplot as plt
import timeit
import argparse

NUMBER_OF_TREES = 50

TARGET_COLUMN = 'flow_size'


def print_performance(inputs, outputs, features, MODEL_SAVE_PATH, op):
    real = []
    predicted = []
    # for f in files:
    #     data = xgboost_util.prepare_files([f], WINDOW_SIZE, scaling, TARGET_COLUMN)
    #     inputs, outputs = xgboost_util.make_io(data)

    model = pickle.load(open(MODEL_SAVE_PATH, "rb"))
    y_pred = model.predict(xgboost.DMatrix(inputs, feature_names=features))
    pred = y_pred.tolist()

    # real += outputs
    # predicted += pred

    xgboost_util.print_metrics(outputs, pred, op)


def main(TEST_NAME, output_file, context, model_train):
    random.seed(0)

    TRAINING_PATH = 'data/ml/' + TEST_NAME + '/training/'
    TEST_PATH = 'data/ml/' + TEST_NAME + '/test/'
    VALIDATION_PATH = 'data/ml/' + TEST_NAME + '/validation/'
    
    if context==True:
        MODEL_SAVE_PATH = 'model/xgboost/with_context/model' + TEST_NAME + '.pkl'
        PLOT_PATH = 'results/xgboost/with_context/plot' + TEST_NAME + '.jpg'
    elif context==False:
        MODEL_SAVE_PATH = 'model/xgboost/without_context/model' + TEST_NAME + '.pkl'
        PLOT_PATH = 'results/xgboost/without_context/plot' + TEST_NAME + '.jpg'
    else:
        pass


    training_files = [os.path.join(TRAINING_PATH, f) for f in os.listdir(TRAINING_PATH)]
    test_files = [os.path.join(TEST_PATH, f) for f in os.listdir(TEST_PATH)]
    validation_files = [os.path.join(VALIDATION_PATH, f) for f in os.listdir(VALIDATION_PATH)]

    scaling = xgboost_util.calculate_scaling(training_files)
    
    if context==True:
        WINDOW_SIZE = 5
        data = xgboost_util.prepare_files(training_files, WINDOW_SIZE, scaling, TARGET_COLUMN)
        data_val = xgboost_util.prepare_files(validation_files, WINDOW_SIZE, scaling, TARGET_COLUMN)
        data_test = xgboost_util.prepare_files(test_files, WINDOW_SIZE, scaling, TARGET_COLUMN)

    elif context==False:
        WINDOW_SIZE = 1
        data = xgboost_util.prepare_files(training_files, WINDOW_SIZE, scaling, TARGET_COLUMN)
        data_val = xgboost_util.prepare_files(validation_files, WINDOW_SIZE, scaling, TARGET_COLUMN)
        data_test = xgboost_util.prepare_files(test_files, WINDOW_SIZE, scaling, TARGET_COLUMN)
    else:
        pass
    
    
    inputs, outputs = xgboost_util.make_io(data)
    inputs_val, outputs_val = xgboost_util.make_io(data_val)
    inputs_test, outputs_test = xgboost_util.make_io(data_test)

    # fit model no training data
    param = {
        'num_epochs': NUMBER_OF_TREES,
        'max_depth': 10,
        'objective': 'reg:linear',
        'booster': 'gbtree',
        'base_score': 2,
        'silent': 1,
        'eval_metric': 'mae'
    }

    if model_train==True:
        training = xgboost.DMatrix(inputs, outputs, feature_names=data[0][0].columns)
        print('Training started')
        model = xgboost.train(param, training, param['num_epochs'])
        print("*** Model fitted ***")
        print("Saved model to disk")
        pickle.dump(model, open(MODEL_SAVE_PATH, "wb"))

    print('TRAINING')
    output_file.write('TRAINING\n')
    print_performance(inputs, outputs, data[0][0].columns, MODEL_SAVE_PATH, output_file)

    print('TEST')
    output_file.write('TEST\n')
    print_performance(inputs_test, outputs_test, data_test[0][0].columns, MODEL_SAVE_PATH, output_file)

    print('VALIDATION')
    output_file.write('VALIDATION\n')
    print_performance(inputs_val, outputs_val, data_val[0][0].columns, MODEL_SAVE_PATH, output_file)

    model = pickle.load(open(MODEL_SAVE_PATH, "rb"))
    plt.figure()
    xgboost.plot_importance(model, importance_type="gain", max_num_features=10)
    plt.rcParams['figure.figsize'] = [30, 10]
    plt.title('XGBoost Feature importance plot for case ' + TEST_NAME)
    plt.savefig(PLOT_PATH)
    # plt.show()


if __name__ == "__main__":
        
    parser = argparse.ArgumentParser()
    parser.add_argument('-train', action="store_true", default=False)   
    args = parser.parse_args()
    
    RESULTS_PATH = 'results/xgboost'
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
    output_file.write('RUNNING MODELS WITHOUT CONTEXT\n\n')
    for test_name in test_names:
        print("Dataset used: %s" %(test_name))
        output_file.write('CASE: '+ test_name + '\n')
        main(test_name, output_file, context=False, model_train=args.train)
        
    output_file.close()
    
