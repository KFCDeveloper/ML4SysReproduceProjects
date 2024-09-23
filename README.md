# Main paper
[https://www.usenix.org/system/files/nsdi19-dukic.pdf] NSDI 19: Is advance knowledge of flow sizes a plausible assumption?

## Traces are obtained from the following applications:
- KMeans
- PageRank
- SGD
All implemented on Spark clusters
- Tensorflow and Web Workload are not tested in our project, due to extremely long time taken by the Tensorflow dataset and also their different characteristics from jobs run on Spark clusters.

## Machine Learning models implemented:
- Feed Forward Neural Networks
- Long Short Term Memory
- Gradient Boosting Decision Trees

## Running our project
To just test the existing models, run `./test_models.sh`.

To train new models and test them, run `./train_models.sh`

All code is in Python3, and can be found in the `ml/` directory.

## Models and Results
- Models are found in `model`, and then under the appropriate subdirectory (FFNN, LSTM or XGBoost). For each approach, we have models trained with and without context.
- Classification models can be found in `model/classification` and are again trained with and without context.
- Results and plots can be found in `results/`, with the same folder structure as the models. 

**Note** : Due to the large size of the files, Git open downloads pointers instead of the original files. You can use `git lfs` to download the complete files.

For further details, please consult our slides and the linked document at the end: https://docs.google.com/presentation/d/1j2QaCtdq03E_P1AwEKAV2JEq9szJd-oHqz1DRmF1veA/edit?usp=sharing
