# coding=utf-8

# we are going to test the ability of generalization
# we change the number of the devices and the nodes


import warnings
warnings.filterwarnings("ignore")
from scipy import stats
import pandas as pd
import numpy as np
# %load_ext autoreload
# %autoreload 2
from code_deepQueueNet.config import BaseConfig, modelConfig
from code_deepQueueNet.util import inference
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('ticks')


def test_0():

    import tqdm
    import matplotlib.patches as mpatches
    import pandas as pd
    import scipy.stats as measures



    def showFig(df):
        for col in ['delay', 'jitter']:
            fig, (ax) = plt.subplots(figsize=(8, 5))
            a_val = 1.
            colors = ['r', 'b', 'g']
            circ1 = mpatches.Patch(edgecolor=colors[0], alpha=a_val, linestyle='-', label='MAP - sim', fill=False)
            circ2 = mpatches.Patch(edgecolor=colors[0], alpha=a_val, linestyle='--', label='MAP - prediction',
                                   fill=False)
            circ3 = mpatches.Patch(edgecolor=colors[1], alpha=a_val, linestyle='-', label='Poisson - sim', fill=False)
            circ4 = mpatches.Patch(edgecolor=colors[1], alpha=a_val, linestyle='--', label='Poisson - prediction',
                                   fill=False)
            circ5 = mpatches.Patch(edgecolor=colors[2], alpha=a_val, linestyle='-', label='Onoff - sim', fill=False)
            circ6 = mpatches.Patch(edgecolor=colors[2], alpha=a_val, linestyle='--', label='Onoff - prediction',
                                   fill=False)

            for i, c in zip(['MAP', 'Poisson', 'Onoff'], colors):
                bins = np.histogram(np.hstack((df[df.tp == i][col + '_sim'].values,
                                               df[df.tp == i][col + '_pred'].values)), bins=100)[1]
                plt.hist(df[df.tp == i][col + '_sim'].values, bins, density=True, color=c, histtype='step',
                         linewidth=1.5);
                plt.hist(df[df.tp == i][col + '_pred'].values, bins, density=True, color=c, histtype='step',
                         linestyle='--', linewidth=1.5);
            ax.legend(handles=[circ1, circ2, circ3, circ4, circ5, circ6], loc=1, fontsize=14)
            plt.xlabel(col.capitalize() + ' (sec)', fontsize=14)
            plt.ylabel('PDF', fontsize=14)
            plt.tick_params(labelsize=12)

            fig, (ax) = plt.subplots(figsize=(8, 5))
            for i, c in zip(['MAP', 'Poisson', 'Onoff'], colors):
                res = stats.relfreq(df[df.tp == i][col + '_sim'].values, numbins=100)
                x = res.lowerlimit + np.linspace(0, res.binsize * res.frequency.size, res.frequency.size)
                y = np.cumsum(res.frequency)
                plt.plot(x, y, color=c, linewidth=1.5)
                res = stats.relfreq(df[df.tp == i][col + '_pred'].values, numbins=100)
                x = res.lowerlimit + np.linspace(0, res.binsize * res.frequency.size, res.frequency.size)
                y = np.cumsum(res.frequency)
                plt.plot(x, y, color=c, linestyle='--', linewidth=1.5)
            plt.xlabel(col.capitalize() + ' (sec)', fontsize=14)
            plt.ylabel('CDF', fontsize=14)
            ax.legend(handles=[circ1, circ2, circ3, circ4, circ5, circ6], loc=4, fontsize=14)
            plt.tick_params(labelsize=12)

    def mergeTrace():
        result = pd.DataFrame()
        for traffic_pattern in tqdm.tqdm(['onoff', 'poisson', 'map']):
            for filename in ['rsim1', 'rsim2', 'rsim3', 'rsim4', 'rsim5']:
                t = pd.read_csv('./data/fattree16/{}/{}_pred.csv'.format(traffic_pattern, filename))
                t['delay_sim'] = t['dep_time'] - t['timestamp (sec)']  # dep_time 是 ground truth
                t['delay_pred'] = t['etime'] - t['timestamp (sec)']  # etime 是 dqn 算出来的
                t['fd'] = t['path'].apply(lambda x: len(x.split('-')))
                t['jitter_sim'] = t.groupby(['src_port', 'path'])['delay_sim'].diff().abs()
                t['jitter_pred'] = t.groupby(['src_port', 'path'])['delay_pred'].diff().abs()
                if traffic_pattern == 'map':
                    t['tp'] = 'MAP'
                else:
                    t['tp'] = traffic_pattern.capitalize()
                result = pd.concat([result, t], ignore_index=True)
        return result

    result = mergeTrace()
    showFig(result.dropna())

if __name__ == '__main__':
    traffic_pattern = 'map'
    config = BaseConfig()
    model_config = modelConfig()
    for i in range(5):  # 5
        ins = inference.INFER('./data/fattree16/{}/rsim{}'.format(traffic_pattern, i + 1),
                              config,
                              model_config)
        ins.run(gpu_number=4)  # please set the value of gpu_number (1,2,4) accordingly.
    test_0()
