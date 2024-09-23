#!/bin/bash
echo "Training the models for prediction"

echo -e "\n\n++++++++++++  Feed Forward Neural Network  ++++++++++++"
python3 ml/ffnn.py -train

echo -e "\n\n++++++++++++  Long Short Term Memory  ++++++++++++"
python3 ml/lstm.py -train

echo -e "\n\n++++++++++++  XGBoost  ++++++++++++"
python3 ml/xgboost_learn.py -train
