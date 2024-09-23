import xgboost
import os
import xgboost_util
import shap
import random
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score
from sklearn.metrics import r2_score
import pickle
import argparse
from show_plots import my_confusion_matrix

NUMBER_OF_TREES = 50

TARGET_COLUMN = 'flow_size'

def prepare_files_classify(files, window_size, scaling, target_column='flow_size'):
    result = []

    for f in files:
        df = pd.read_csv(f, index_col=False)

        df = df.drop("index", axis=1)

        flow_size = df[target_column]
        thresh = 1e6
        flow_size = np.where(flow_size >= thresh, 1, -1)
        
        df = df.apply((lambda x: xgboost_util.resize(x, scaling)), axis=0)
        df[target_column] = flow_size
        
        #extend the window
        columns = list(df)
        final_df = df.copy()
        for sample_num in range(1, window_size):
            shifted = df.shift(sample_num)
            shifted.columns = map(lambda x: x+str(sample_num), shifted.columns)
            final_df = pd.concat([shifted, final_df], axis=1)

        final_df = final_df.fillna(0)
        final_df = final_df.drop(target_column, axis=1)

        result.append((final_df, flow_size))

    return result

def print_performance(eval_type, inputs, outputs, MODEL_SAVE_PATH, PLOT_PATH, op):
    real = []
    predicted = []
    # for f in files:
    #     data = xgboost_util.prepare_files([f], WINDOW_SIZE, scaling, TARGET_COLUMN)
    #     inputs, outputs = xgboost_util.make_io(data)

    model = pickle.load(open(MODEL_SAVE_PATH, "rb"))
    pred = model.predict(inputs)
    
    accuracy = accuracy_score(outputs, pred)
    print("Accuracy: %f" % accuracy)
    op.write("Accuracy: %f" % accuracy)
    print("R2: %f" % r2_score(outputs, pred))
    op.write("\nR2: %f\n" % r2_score(outputs, pred))
    
    if eval_type == 'test':
        df_cm = confusion_matrix(outputs, pred)
        my_confusion_matrix(df_cm, PLOT_PATH)


def main(TEST_NAME, output_file, context, model_train, ml_model):
    random.seed(0)

    TRAINING_PATH = 'data/ml/' + TEST_NAME + '/training/'
    TEST_PATH = 'data/ml/' + TEST_NAME + '/test/'
    VALIDATION_PATH = 'data/ml/' + TEST_NAME + '/validation/'
    
    if context==True:
        if ml_model == 'RandomForest':
            MODEL_SAVE_PATH = 'model/classification/with_context/model_rf_'+ TEST_NAME + '.pkl'
            PLOT_PATH = 'results/classification/with_context/plot_rf_' + TEST_NAME + '.jpg'
        if ml_model == 'FFNN':
            MODEL_SAVE_PATH = 'model/classification/with_context/model_ffnn_'+ TEST_NAME + '.pkl'
            PLOT_PATH = 'results/classification/with_context/plot_ffnn_' + TEST_NAME + '.jpg'
    
    elif context==False:
        if ml_model == 'RandomForest':
            MODEL_SAVE_PATH = 'model/classification/without_context/model_rf_'+ TEST_NAME + '.pkl'
            PLOT_PATH = 'results/classification/without_context/plot_rf_' + TEST_NAME + '.jpg'
        if ml_model == 'FFNN':
            MODEL_SAVE_PATH = 'model/classification/without_context/model_ffnn_'+ TEST_NAME + '.pkl'
            PLOT_PATH = 'results/classification/without_context/plot_ffnn_' + TEST_NAME + '.jpg'
    else:
        pass


    training_files = [os.path.join(TRAINING_PATH, f) for f in os.listdir(TRAINING_PATH)]
    test_files = [os.path.join(TEST_PATH, f) for f in os.listdir(TEST_PATH)]
    validation_files = [os.path.join(VALIDATION_PATH, f) for f in os.listdir(VALIDATION_PATH)]

    scaling = xgboost_util.calculate_scaling(training_files)
    
    if context==True:
        WINDOW_SIZE = 5
        data = prepare_files_classify(training_files, WINDOW_SIZE, scaling, TARGET_COLUMN)
        data_val = prepare_files_classify(validation_files, WINDOW_SIZE, scaling, TARGET_COLUMN)
        data_test = prepare_files_classify(test_files, WINDOW_SIZE, scaling, TARGET_COLUMN)

    elif context==False:
        WINDOW_SIZE = 1
        data = prepare_files_classify(training_files, WINDOW_SIZE, scaling, TARGET_COLUMN)
        data_val = prepare_files_classify(validation_files, WINDOW_SIZE, scaling, TARGET_COLUMN)
        data_test = prepare_files_classify(test_files, WINDOW_SIZE, scaling, TARGET_COLUMN)
    else:
        pass
    
    
    inputs, outputs = xgboost_util.make_io(data)
    inputs_val, outputs_val = xgboost_util.make_io(data_val)
    inputs_test, outputs_test = xgboost_util.make_io(data_test)

    if model_train==True:
        
        if ml_model == 'RandomForest':
            rf = RandomForestClassifier(n_estimators = NUMBER_OF_TREES, class_weight='balanced')
            model = rf.fit(inputs, outputs)
            pickle.dump(model, open(MODEL_SAVE_PATH, "wb"))
            feature_importances = pd.DataFrame(rf.feature_importances_,
                                   index = data[0][0].columns,
                                    columns=['importance']).sort_values('importance', ascending=False)
            #print(feature_importances)
            print("****Model fitted****")
            
        if ml_model == 'FFNN':
            mlp = MLPClassifier(hidden_layer_sizes=(5, 5), activation='relu')
            model = mlp.fit(inputs, outputs)
            pickle.dump(model, open(MODEL_SAVE_PATH, "wb"))
            print("****Model fitted****")


    print('TRAINING')
    
    output_file.write('TRAINING\n')
    print_performance('train', inputs, outputs, MODEL_SAVE_PATH, PLOT_PATH, output_file)

    print('TEST')
    output_file.write('TEST\n')
    print_performance('test', inputs_test, outputs_test, MODEL_SAVE_PATH, PLOT_PATH, output_file)

    print('VALIDATION')
    output_file.write('VALIDATION\n')
    print_performance('validation', inputs_val, outputs_val, MODEL_SAVE_PATH, PLOT_PATH, output_file)


if __name__ == "__main__":
        
    parser = argparse.ArgumentParser()
    parser.add_argument('-train', action="store_true", default=False)   
    args = parser.parse_args()
    
    RESULTS_PATH = 'results/classification'
    output_file = open(os.path.join(RESULTS_PATH, 'results.txt'), 'w+')
    test_names = ["KMeans", "PageRank", "SGD"]
    ml_models = ['RandomForest', 'FFNN']
    print("Running all experiments:\n")

    print("***********Running models with context***********")
    output_file.write('RUNNING MODELS WITH CONTEXT\n\n')
    for ml_model in ml_models:
        print("\nMODEL: %s" %(ml_model))
        output_file.write('\nMODEL: '+ ml_model + '\n')
        for test_name in test_names:
            print("\nCase %s" %(test_name))
            output_file.write('\nCASE: '+ test_name + '\n')
            main(test_name, output_file, context=True, model_train=args.train, ml_model=ml_model)
    
    print("***********Running models without context***********")
    output_file.write('\n\nRUNNING MODELS WITHOUT CONTEXT\n\n')
    for ml_model in ml_models:
        print("\nMODEL: %s" %(ml_model))
        output_file.write('\n\nMODEL: '+ ml_model + '\n')
        for test_name in test_names:
            print("\nCase %s" %(test_name))
            output_file.write('\nCASE: '+ test_name + '\n')
            main(test_name, output_file, context=False, model_train=args.train, ml_model=ml_model)
        
    output_file.close()