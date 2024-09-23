# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 15:57:38 2020

@author: shreya
"""
import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sn

def ml_plots(filename, dataset):
    hist = pickle.load(open(filename, "rb" ) )
    SAVE_PLOT= os.path.dirname(filename) + '/' + dataset + '.jpg'
    
    val_loss = hist['val_loss']
    loss = hist['loss']
    
    plt.figure()
    plt.plot(val_loss, label='Validation Loss')
    plt.plot(loss, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.xticks(np.arange(0, max(len(val_loss), len(loss))+1, 10))
    plt.title(('Losses over epoch: ' + dataset))
    plt.legend(loc= 'upper right', bbox_to_anchor=(1.1, 1.05))
    
    plt.savefig(SAVE_PLOT)
    #plt.show()

def my_confusion_matrix(df_cm, SAVE_PATH):
    
    plt.figure()
    group_names = ['True Neg','False Pos','False Neg','True Pos']
    group_counts = ["{0:0.0f}".format(value) for value in
                    df_cm.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in
                         df_cm.flatten()/np.sum(df_cm)]
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
              zip(group_names,group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)

    sn.heatmap(df_cm, annot=labels, fmt='', cmap='Blues')
    plt.savefig(SAVE_PATH)
    #plt.show()