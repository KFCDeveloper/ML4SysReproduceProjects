import os
import sys
print("Script started")
from sklearn.preprocessing import MinMaxScaler
cwd = os.getcwd()
assert "networking_envs" in cwd
sys.path.append(cwd[:cwd.find("networking_envs")] + "networking_envs")
sys.path.append(cwd[:cwd.find("networking_envs")] + "openai_baselines")
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import random_split
from scipy.linalg import eigh
from torch import nn
from torch.utils.data import Dataset, DataLoader
from networking_envs.networking_env.environments.ecmp.env_args_parse import parse_args
from networking_envs.networking_env.environments.ecmp import history_env
from networking_envs.networking_env.environments.consts import SOMode
from networking_envs.networking_env.utils.shared_consts import SizeConsts
from tqdm import tqdm
from networking_envs.meta_learning.meta_const import RNN_Cons, DOTE_Cons
from torch.utils.data import Subset
from const_dote import *
import scipy.io
import scipy.linalg
import sklearn.metrics
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import sklearn.metrics
from scipy.sparse.linalg import eigs
import sklearn.metrics
from scipy.linalg import eigh
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

from sklearn.base import BaseEstimator
from scipy.sparse.linalg import eigs
from scipy.spatial.distance import cdist as sp_cdist

class TCA:
    def __init__(self, kernel_type='linear', dim=30, lamb=1, gamma=1):
        self.kernel_type = kernel_type
        self.dim = dim
        self.lamb = lamb
        self.gamma = gamma
        self.scaler = MinMaxScaler()

    def kernel(self, ker, X1, X2=None):
        if ker == 'primal':
            return X1
        elif ker == 'linear':
            return sklearn.metrics.pairwise.linear_kernel(X1, X2) if X2 is not None else sklearn.metrics.pairwise.linear_kernel(X1)
        elif ker == 'rbf':
            return sklearn.metrics.pairwise.rbf_kernel(X1, X2, self.gamma) if X2 is not None else sklearn.metrics.pairwise.rbf_kernel(X1, None, self.gamma)
        elif ker == 'gaussian':
            if X2 is None:
                X2 = X1
            pairwise_sq_dists = euclidean_distances(X1, X2, squared=True)
            return np.exp(-pairwise_sq_dists / (2 * (self.gamma ** 2)))
        else:
            raise ValueError(f"Unknown kernel type: {self.kernel_type}")

        
    def fit(self, Xs, Xt):
        X = np.hstack((Xs.T, Xt.T))
        m, n = X.shape
        ns, nt = len(Xs), len(Xt)
        e = np.vstack((1 / ns * np.ones((ns, 1)), -1 / nt * np.ones((nt, 1))))
        M = e @ e.T
        M = M / np.linalg.norm(M, 'fro')
        H = np.eye(n) - 1 / n * np.ones((n, n))
        K = self.kernel(self.kernel_type, X, None)
        n_eye = m if self.kernel_type == 'primal' else n
        a, b = K @ M @ K.T + self.lamb * np.eye(n_eye), K @ H @ K.T

        # Use `eigh` instead of `eig` for real symmetric matrices
        w, V = scipy.linalg.eigh(a, b)
        ind = np.argsort(w)
        A = V[:, ind[:self.dim]]
        Z = A.T @ K
        Z /= np.linalg.norm(Z, axis=0)
        print("TCA done finally")
        Xs_new, Xt_new = Z[:, :ns].T, Z[:, ns:].T
        return Xs_new, Xt_new

    
# dataset definition
class DmDataset(Dataset):
    def __init__(self, props=None, env=None, is_test=None):
        # store the inputs and outputs
        assert props != None and env != None and is_test != None

        num_nodes = env.get_num_nodes()
        env.test(is_test)
        tms = env._simulator._cur_hist._tms
        if props.ecmp_topo in FACTOR_MAP:
            tms = [x * FACTOR_MAP[props.ecmp_topo] for x in tms]
        opts = env._simulator._cur_hist._opts
        tms = [np.asarray([tms[i]]) for i in range(len(tms))]
        np_tms = np.vstack(tms)
        np_tms = np_tms.T
        np_tms_flat = np_tms.flatten('F')
        assert (len(tms) == len(opts))
        X_ = []
        for histid in range(len(tms) - props.hist_len):
            start_idx = histid * num_nodes * (num_nodes - 1)
            end_idx = start_idx + props.hist_len * num_nodes * (num_nodes - 1) # hist_len default=12, this assumes that we sample 12 TMs per hour (1 per 5min) 
            X_.append(np_tms_flat[start_idx:end_idx])

        self.X = np.asarray(X_)
        self.y = np.asarray([np.append(tms[i], opts[i]) for i in range(props.hist_len, len(opts))]) # 这里难以理解，这个tms[i]只是props.hist_len后的流量矩阵啊，放这里是什么意思


    # number of rows in the dataset
    def __len__(self):
        return len(self.X)

    # get a row at an index
    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]


# dataset definition
class TCADmDataset(Dataset):
    def __init__(self, source_props=None, target_props=None, is_test=None, tca_flag = False, target_env = None, source_env=None):
        # store the inputs and outputs
        assert target_props != None and target_env != None and is_test != None

        num_nodes = target_env.get_num_nodes()
        target_env.test(is_test)
        target_tms = target_env._simulator._cur_hist._tms

        source_env.test(is_test)
        source_tms = source_env._simulator._cur_hist._tms

        if target_props.ecmp_topo in FACTOR_MAP:
            target_tms = [x * FACTOR_MAP[target_props.ecmp_topo] for x in target_tms]
        if source_props.ecmp_topo in FACTOR_MAP:
            source_tms = [x * FACTOR_MAP[source_props.ecmp_topo] for x in source_tms]
            
        if tca_flag:
            source = np.asarray(source_tms)
            target = np.asarray(target_tms)
            tca = TCA(kernel_type='primal', dim=source.shape[1], lamb=0.5, gamma=0.5)
            _, tca_tms = tca.fit(source, target)
            tca_tms = ((tca_tms - tca_tms.min()) / (tca_tms.max() - tca_tms.min())) * (target.max() - target.min()) + target.min() # transform
        

        opts = target_env._simulator._cur_hist._opts


        tca_tms = [np.asarray([tca_tms[i]]) for i in range(len(tca_tms))]  # TODO `*1e-8`
        np_tms = np.vstack(tca_tms)
        np_tms = np_tms.T
        np_tms_flat = np_tms.flatten('F')
        
        
        # TCA insertion
        assert (len(tca_tms) == len(opts))
        X_ = []
        for histid in range(len(tca_tms) - target_props.hist_len):
            start_idx = histid * num_nodes * (num_nodes - 1)
            end_idx = start_idx + target_props.hist_len * num_nodes * (num_nodes - 1) # hist_len default=12, this assumes that we sample 12 TMs per hour (1 per 5min) 
            X_.append(np_tms_flat[start_idx:end_idx])

        self.X = np.asarray(X_)
        self.y = np.asarray([np.append(tca_tms[i], opts[i]) for i in range(props.hist_len, len(opts))]) # 这里难以理解，这个tms[i]只是props.hist_len后的流量矩阵啊，放这里是什么意思


    # number of rows in the dataset
    def __len__(self):
        return len(self.X)

    # get a row at an index
    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]


# define fixed input and output dimension model. to adapt all the input size.
class AdaptiveNeuralNetworkMaxUtil(nn.Module):
    def __init__(self, input_dim, output_dim,node_num):
        super(AdaptiveNeuralNetworkMaxUtil, self).__init__()
        # FC layer for input size adapting
        self.input_main_layer = nn.Linear(node_num*(node_num-1) * DOTE_Cons.HIST_LEN, 1320)
        self.flatten = nn.Flatten()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),    # 我隔着儿多加了一层 (包含nn.ReLu)
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Sigmoid()    # 这里用sigmod是因为，比如输出 a b的所有path上的百分比，因为已知了a到b的demand，就能算每个path上面flow的bw了
        )

    def forward(self, input_):
        main_model_input = self.input_main_layer(input_)
        x = self.flatten(main_model_input)
        logits = self.net(x)
        return logits

class NeuralNetworkMaxFlowMaxConc(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(NeuralNetworkMaxFlowMaxConc, self).__init__()
        self.flatten = nn.Flatten()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, output_dim),
            # nn.ELU(alpha=0.1)
            nn.Sigmoid()
            # nn.Softmax(dim=1)
            # nn.ReLU()
        )
    
    def forward(self, x):
        x = self.flatten(x)
        logits = self.net(x)
        # for layer in self.net[:-1]:  # Skip the last layer
        #     x = layer(x)
        #  # Save the input to the last layer
        # input_to_last_layer = x.clone()  # Clone to avoid in-place modifications
        
        # # Pass through the final layer
        # logits = self.net[-1](x)
        return logits 

    
def loss_fn_maxflow_maxconc(y_pred_batch, y_true_batch, env):
    num_nodes = env.get_num_nodes()

    losses = []
    loss_vals = []
    batch_size = y_pred_batch.shape[0]
    
    for i in range(batch_size):
        y_pred = y_pred_batch[[i]]
        y_true = y_true_batch[[i]]
        opt = y_true[0][num_nodes * (num_nodes - 1)].item()
        y_true = torch.narrow(y_true, 1, 0, num_nodes * (num_nodes - 1))    # 这里的y_true 应该是指不调度，这个his_len后的那个带宽分布（根据他的输入构造，得出的看法）

        y_pred = y_pred + 0.1 #ELU # ydy : pred 应该是正数才对
        # y_pred 本来是 基于 paths的（并且预测出来应该是百分比，一对节点之间的带宽，分到这个节点之间所有路径上，每个路径带宽占总的百分比）
        # ，然后转化为edges(不是隧道)上的带宽占比（！！不是带宽值，是占比）; edge是边；commodity是 每两对节点
        edges_weight = paths_to_edges.transpose(0,1).matmul(torch.transpose(y_pred, 0, 1))  
        alpha = torch.max(edges_weight.divide(torch.tensor(np.array([env._capacities])).to(device).transpose(0,1))) # 所有edge上 带宽占比最大的那个;算出来是值是 weight/cap 并不是 带宽/cap
        max_flow_on_tunnel = y_pred / alpha     # 可能也通过这里的调控，使得分配带宽不大于 cap # alpha类似于放缩，要把pred的值放缩到和 y_true差不多的范围；但不太明白，为什么要算这个alpha，搞这个放缩，很奇怪
        # 我把上面三行注释掉了，实在不知道有啥用; 关于上面这行，确实有调控 pred 不要大于cap的作用应该
        # max_flow_on_tunnel = y_pred
        max_flow_per_commodity = commodities_to_paths.matmul(max_flow_on_tunnel.transpose(0,1)) # 基于path的带宽转为基于 隧道（不是edge）的带宽 # commodities_to_paths 是一个0,1 的矩阵，应该是 一对节点之间，会经过哪些path，经过的标为1

        if props.opt_function == "MAXFLOW": #MAX FLOW *(1e10)
            max_mcf = torch.sum(torch.minimum(max_flow_per_commodity.transpose(0,1), y_true))  # 这个的意思有点像concurrent obj啊，这里是希望每个commodity的flow都尽量大，至少要比这个不调度的大（但我觉得还是有问题，因为我可以牺牲某一个commodity让整体更大） # 应该是要让隧道的带宽越大越好（因为可能因为链路容量问题，永远无法完全分配完） # 应该是和 y_true 一个性质的；表示计算出来的 每个隧道上的带宽
            
            # loss = -max_mcf if max_mcf.item() == 0.0 else -max_mcf/max_mcf.item()   # previous one
            # loss = -max_mcf + torch.sum(y_true) # *(1e10)
            loss = opt - SizeConsts.GBPS_TO_BPS(max_mcf) 
            # loss_val = 1.0 if opt == 0.0 else max_mcf.item()/SizeConsts.BPS_TO_GBPS(opt)
            loss_val = 1.0 if opt == 0.0 else SizeConsts.GBPS_TO_BPS(max_mcf.item())/opt  # ydy: *(1e4), 我也不知道为什么差这么多
        
        
        elif props.opt_function == "MAXCONC": #MAX CONCURRENT FLOW
            actual_flow_per_commodity = torch.minimum(max_flow_per_commodity.transpose(0,1), y_true)
            max_concurrent_vec = torch.full_like(actual_flow_per_commodity, fill_value=1.0)
            mask = y_true != 0
            max_concurrent_vec[mask] = actual_flow_per_commodity[mask].divide(y_true[mask])
            max_concurrent = torch.min(max_concurrent_vec)
            
            #actual_flow_per_commodity = torch.minimum(max_flow_per_commodity.transpose(0,1), y_true)
            #actual_flow_per_commodity = torch.maximum(actual_flow_per_commodity, torch.tensor([1e-32]))
            #max_concurrent = torch.min(actual_flow_per_commodity.divide(torch.maximum(y_true, torch.tensor([1e-32])))
            
            loss = -max_concurrent if max_concurrent.item() == 0.0 else -max_concurrent/max_concurrent.item()
            loss_val = 1.0 if opt == 0.0 else max_concurrent.item()/opt
                
            #update concurrent flow statistics
            if concurrent_flow_cdf != None:
                curr_dm_conc_flow_cdf = [0]*len(concurrent_flow_cdf)
                for j in range(env.get_num_nodes() * (env.get_num_nodes() - 1)):
                    allocated = max_flow_per_commodity[j][0].item()
                    actual = y_true[0][j].item()
                    curr_dm_conc_flow_cdf[j] = 1.0 if actual == 0 else min(1.0, allocated / actual)
                curr_dm_conc_flow_cdf.sort()
                
                for j in range(len(curr_dm_conc_flow_cdf)):
                    concurrent_flow_cdf[j] += curr_dm_conc_flow_cdf[j]
        else:
            assert False
        
        losses.append(loss)
        loss_vals.append(loss_val)

    ret = sum(losses) / len(losses)
    ret_val = sum(loss_vals) / len(loss_vals)
    
    return ret, ret_val


def loss_fn_maxflow_maxconc_target(y_pred_batch, y_true_batch, env_base):
    num_nodes = env_base.get_num_nodes()

    losses = []
    loss_vals = []
    batch_size = y_pred_batch.shape[0]
    
    for i in range(batch_size):
        y_pred = y_pred_batch[[i]]
        y_true = y_true_batch[[i]]
        opt = y_true[0][num_nodes * (num_nodes - 1)].item()
        y_true = torch.narrow(y_true, 1, 0, num_nodes * (num_nodes - 1))    # 这里的y_true 应该是指不调度，这个his_len后的那个带宽分布（根据他的输入构造，得出的看法）

        y_pred = y_pred + 0.1 #ELU # ydy : pred 应该是正数才对
        # y_pred 本来是 基于 paths的（并且预测出来应该是百分比，一对节点之间的带宽，分到这个节点之间所有路径上，每个路径带宽占总的百分比）
        # ，然后转化为edges(不是隧道)上的带宽占比（！！不是带宽值，是占比）; edge是边；commodity是 每两对节点
        edges_weight = paths_to_edges_base.transpose(0,1).matmul(torch.transpose(y_pred, 0, 1))  
        alpha = torch.max(edges_weight.divide(torch.tensor(np.array([env_base._capacities])).to(device).transpose(0,1))) # 所有edge上 带宽占比最大的那个;算出来是值是 weight/cap 并不是 带宽/cap
        max_flow_on_tunnel = y_pred / alpha     # 可能也通过这里的调控，使得分配带宽不大于 cap # alpha类似于放缩，要把pred的值放缩到和 y_true差不多的范围；但不太明白，为什么要算这个alpha，搞这个放缩，很奇怪
        # 我把上面三行注释掉了，实在不知道有啥用; 关于上面这行，确实有调控 pred 不要大于cap的作用应该
        # max_flow_on_tunnel = y_pred
        max_flow_per_commodity = commodities_to_paths.matmul(max_flow_on_tunnel.transpose(0,1)) # 基于path的带宽转为基于 隧道（不是edge）的带宽 # commodities_to_paths 是一个0,1 的矩阵，应该是 一对节点之间，会经过哪些path，经过的标为1

        if props.opt_function == "MAXFLOW": #MAX FLOW *(1e10)
            max_mcf = torch.sum(torch.minimum(max_flow_per_commodity.transpose(0,1), y_true))  # 这个的意思有点像concurrent obj啊，这里是希望每个commodity的flow都尽量大，至少要比这个不调度的大（但我觉得还是有问题，因为我可以牺牲某一个commodity让整体更大） # 应该是要让隧道的带宽越大越好（因为可能因为链路容量问题，永远无法完全分配完） # 应该是和 y_true 一个性质的；表示计算出来的 每个隧道上的带宽
            
            # loss = -max_mcf if max_mcf.item() == 0.0 else -max_mcf/max_mcf.item()   # previous one
            # loss = -max_mcf + torch.sum(y_true) # *(1e10)
            # loss = opt - max_mcf*(1e4)
          
            # # loss_val = 1.0 if opt == 0.0 else max_mcf.item()/SizeConsts.BPS_TO_GBPS(opt)
            # loss_val = 1.0 if opt == 0.0 else max_mcf.item()*(1e4)/opt  # ydy: *(1e4), 我也不知道为什么差这么多
            loss = opt - SizeConsts.GBPS_TO_BPS(max_mcf) 
            # loss_val = 1.0 if opt == 0.0 else max_mcf.item()/SizeConsts.BPS_TO_GBPS(opt)
            loss_val = 1.0 if opt == 0.0 else SizeConsts.GBPS_TO_BPS(max_mcf.item())/opt  # ydy: *(1e4), 我也不知道为什么差这么多
    
        
        losses.append(loss)
        loss_vals.append(loss_val)

    ret = sum(losses) / len(losses)
    ret_val = sum(loss_vals) / len(loss_vals)
    
    return ret, ret_val



    
def choose_part_dataset(portion,input_dataset):
    # 假设你的 input_dataset 已经定义
    dataset_size = len(input_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(dataset_size * portion))  # 一部分的数据
    np.random.seed(42)
    np.random.shuffle(indices)
    subset_indices = indices[:split]

    # 创建子集
    return Subset(input_dataset, subset_indices)


#############################################################################################################################################################################################
#### tca two datasets
#############################################################################################################################################################################################

# Dataset 1, source domain from arg
props = parse_args(sys.argv[1:])
env = history_env.ECMPHistoryEnv(props)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

ctp_coo = env._optimizer._commodities_to_paths.tocoo()
commodities_to_paths = torch.sparse_coo_tensor(np.vstack((ctp_coo.row, ctp_coo.col)), torch.DoubleTensor(ctp_coo.data), torch.Size(ctp_coo.shape)).to(device)
pte_coo = env._optimizer._paths_to_edges.tocoo()
paths_to_edges = torch.sparse_coo_tensor(np.vstack((pte_coo.row, pte_coo.col)), torch.DoubleTensor(pte_coo.data), torch.Size(pte_coo.shape)).to(device)

# Dataset 2, target domain 
props_base = parse_args(['--ecmp_topo', "Abilene-obj2-2-('5', '8')-('6', '7')", '--paths_from', 'sp', '--so_mode', 'train', '--so_epochs', '20', '--so_batch_size', '13', '--opt_function', 'MAXFLOW'])
env_base = history_env.ECMPHistoryEnv(props_base)
ctp_coo_base = env_base._optimizer._commodities_to_paths.tocoo()
commodities_to_paths_base = torch.sparse_coo_tensor(np.vstack((ctp_coo_base.row, ctp_coo_base.col)), torch.DoubleTensor(ctp_coo_base.data), torch.Size(ctp_coo_base.shape)).to(device)
pte_coo_base = env_base._optimizer._paths_to_edges.tocoo()
paths_to_edges_base = torch.sparse_coo_tensor(np.vstack((pte_coo_base.row, pte_coo_base.col)), torch.DoubleTensor(pte_coo_base.data), torch.Size(pte_coo_base.shape)).to(device)



# def test_model_on_test_test(model, test_test_dl):
#     model.eval()  # Set model to evaluation mode
#     test_loss = 0
#     total = 0

#     with torch.no_grad():  # Disable gradient computation for evaluation
#         for inputs, targets in test_test_dl:
#             inputs = inputs.to(device)
#             targets = targets.to(device)
#             outputs = model(inputs)
#             loss, loss_val = loss_fn_maxflow_maxconc_target(outputs, targets, env_base)

#             test_loss += loss.item()
#             # Assuming binary classification or regression; adjust for your task
#             # Example: correct += (outputs.argmax(1) == targets).type(torch.float).sum().item()
#             total += targets.size(0)

#     avg_test_loss = test_loss / len(test_test_dl)
#     print(f"Test set: Average loss: {avg_test_loss:.4f}")
#     model.train()  # Set model back to training mode

def test_model_on_test_test(model, test_test_dl):
    model.eval()  # Set model to evaluation mode
    test_loss = 0
    total = 0
    ema_loss_val = None  # Initialize EMA for loss_val
    alpha = 0.1  # Smoothing factor, adjust as needed (0 < alpha <= 1)

    with torch.no_grad():  # Disable gradient computation for evaluation
        for inputs, targets in test_test_dl:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss, loss_val = loss_fn_maxflow_maxconc_target(outputs, targets, env_base)

            test_loss += loss.item()
            # Update EMA for loss_val
            if ema_loss_val is None:
                ema_loss_val = loss_val  # Initialize on first value
            else:
                ema_loss_val = alpha * loss_val + (1 - alpha) * ema_loss_val

            total += targets.size(0)

    avg_test_loss = test_loss / len(test_test_dl)
    print(f"Test Loss Val: {ema_loss_val:.4f}")
    model.train()  # Set model back to training mode

def split_dataset(dataset, train_ratio):
    """
    Splits the dataset into two parts: train and test.

    Args:
        dataset: The original dataset to be split.
        train_ratio: The ratio of the dataset to use for training. The rest will be used for testing.
    
    Returns:
        subset_train: The training portion of the dataset.
        subset_test: The testing portion of the dataset.
    """
    dataset_size = len(dataset)
    train_size = int(train_ratio * dataset_size)
    test_size = dataset_size - train_size
    

    # Randomly split the dataset
    subset_train, subset_test = random_split(dataset, [train_size, test_size])

    return subset_train, subset_test


batch_size = props.so_batch_size
n_epochs = props.so_epochs
concurrent_flow_cdf = None
if props.opt_function == "MAXFLOW":
    NeuralNetwork = NeuralNetworkMaxFlowMaxConc
    loss_fn = loss_fn_maxflow_maxconc


if props.so_mode == SOMode.TRAIN:  # train

    # Define datasets
    train_dataset = DmDataset(props, env, False)
    test_dataset = TCADmDataset(source_props = props, target_props = props_base, target_env = env_base, source_env = env, is_test = False, tca_flag = True)
    
    ##################################
    ## Following are used for baseline
    ##################################
    
    # Train without TCA
    # train_dataset = DmDataset(props, env, False)
    # test_dataset = DmDataset(props_base, env_base, False)
    
    # Train from scratch on target domain
    # train_dataset = DmDataset(props_base, env_base, False)
    # test_dataset = DmDataset(props_base, env_base, False)
    
    
    
    #############################################################################################################################################################################################
    ####TODO: AL insertion
    #############################################################################################################################################################################################
    # Define portion and split the test dataset
    portion = 1
    subset_train_X = train_dataset # choose_part_dataset(portion, train_dataset)
    subset_test_X = test_dataset # choose_part_dataset(portion, test_dataset)
         
    
    # subset_test_train_X, subset_test_test_X = subset_train_X, subset_test_X # split_dataset(subset_test_X, 0.7)
    dataset_size = len(subset_test_X)
    train_size = int(0.7 * dataset_size)  # 80% 作为训练集

    # 非随机顺序分割ç
    train_indices = list(range(train_size))
    test_indices = list(range(train_size, dataset_size))

    subset_test_train_X = Subset(subset_test_X, train_indices)
    subset_test_test_X = Subset(subset_test_X, test_indices)

    # Create the model
    model = NeuralNetwork(props.hist_len * env.get_num_nodes() * (env.get_num_nodes() - 1), env._optimizer._num_paths)
    model.double()
    model.to(device)
    
    print(f"Number of samples in train_X: {len(subset_train_X)}")
    print(f"Number of samples in test_train_X: {len(subset_test_train_X)}")


    train_dl = DataLoader(subset_train_X, batch_size=batch_size, shuffle=True)
    test_train_dl = DataLoader(subset_test_train_X, batch_size=batch_size, shuffle=True)
    test_test_dl = DataLoader(subset_test_test_X, batch_size=batch_size, shuffle=True)
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters())
    model.double()
    model.to(device)
    # Train on the train dataset
    for epoch in range(n_epochs):
        with tqdm(train_dl) as tepoch:
            epoch_train_loss = []
            loss_sum = loss_count = 0
            real_loss_sum = 0
            for inputs, targets in tepoch:
                inputs = inputs.to(device)
                targets = targets.to(device)

                tepoch.set_description(f"Epoch {epoch}")
                optimizer.zero_grad()
                yhat = model(inputs)
                loss, loss_val = loss_fn(yhat, targets, env) 
                loss.backward()
                optimizer.step()
                epoch_train_loss.append(loss_val)
                loss_sum += loss_val
                real_loss_sum += loss
                loss_count += 1
                loss_avg = loss_sum / loss_count
                real_loss_avg = real_loss_sum / loss_count
                tepoch.set_postfix(loss=f"loss:{real_loss_avg:.5f} loss_val:{loss_avg:.10f}")


    # Save model after training on train dataset
    torch.save(model, 'model_dote_' + props.ecmp_topo + f"_choose_{portion}_train" + '.pkl')
    print("train on the test set")
    # Continue training on test_train dataset for 50 epochs, evaluating on test_test every 5 epochs
    for epoch in range(50):
        with tqdm(test_train_dl) as tepoch:
            epoch_train_loss = []
            loss_sum = loss_count = 0
            real_loss_sum = 0
            for inputs, targets in tepoch:
                inputs = inputs.to(device)
                targets = targets.to(device)

                tepoch.set_description(f"Epoch {epoch}")
                optimizer.zero_grad()
                yhat = model(inputs)
                loss, loss_val = loss_fn_maxflow_maxconc_target(yhat, targets, env_base)
                loss.backward()
                optimizer.step()
                epoch_train_loss.append(loss_val)
                loss_sum += loss_val
                real_loss_sum += loss
                loss_count += 1
                loss_avg = loss_sum / loss_count
                real_loss_avg = real_loss_sum / loss_count
                tepoch.set_postfix(loss=f"loss:{real_loss_avg:.5f} loss_val:{loss_avg:.10f}")

        # Every 5 epochs, test the model on test_test
        if (epoch + 1) % 1 == 0:
            test_model_on_test_test(model, test_test_dl)

    # Save the final model after all training
    torch.save(model, 'model_dote_' + props.ecmp_topo + f"_choose_{portion}_final" + '.pkl')
