# coding=utf-8
import warnings
warnings.filterwarnings("ignore")
# %load_ext autoreload
# %autoreload 2
from code_deepQueueNet import deviceModel
from code_deepQueueNet.config import BaseConfig
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '3'

config = BaseConfig()
model = deviceModel.deepQueueNet(config,
                                 target=['time_in_sys'],
                                 data_preprocessing=True)  # please turn it on when you run the cell for the first time
model.build_and_training()
# 1.30 ~