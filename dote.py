import sys
import os

cwd = os.getcwd()
assert "networking_envs" in cwd
sys.path.append(cwd[:cwd.find("networking_envs")] + "networking_envs")
sys.path.append(cwd[:cwd.find("networking_envs")] + "openai_baselines")

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from networking_envs.networking_env.environments.ecmp.env_args_parse import parse_args
from networking_envs.networking_env.environments.ecmp import history_env
from networking_envs.networking_env.environments.consts import SOMode
from networking_envs.networking_env.utils.shared_consts import SizeConsts
from tqdm import tqdm
from networking_envs.meta_learning.meta_const import RNN_Cons, DOTE_Cons

# dataset definition
class DmDataset(Dataset):
    def __init__(self, props=None, env=None, is_test=None):
        # store the inputs and outputs
        assert props != None and env != None and is_test != None

        num_nodes = env.get_num_nodes()
        env.test(is_test)
        tms = env._simulator._cur_hist._tms
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
        self.y = np.asarray([np.append(tms[i], opts[i]) for i in range(props.hist_len, len(opts))])


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

def loss_fn_maxutil(y_pred_batch, y_true_batch, env):
    num_nodes = env.get_num_nodes()
    
    losses = []
    loss_vals = []
    batch_size = y_pred_batch.shape[0]
    
    for i in range(batch_size):
        y_pred = y_pred_batch[[i]]
        y_true = y_true_batch[[i]]
        opt = y_true[0][num_nodes * (num_nodes - 1)].item()
        y_true = torch.narrow(y_true, 1, 0, num_nodes * (num_nodes - 1))

    
        y_pred = y_pred + 1e-16 #eps
        paths_weight = torch.transpose(y_pred, 0, 1)
        commodity_total_weight = commodities_to_paths.matmul(paths_weight)  # 这个 [110,218] * [218,1] 就是把每个path上的flow的bw 转为 两两节点之间的 bw
        commodity_total_weight = 1.0 / (commodity_total_weight)
        paths_over_total = commodities_to_paths.transpose(0,1).matmul(commodity_total_weight)   # commodities_to_paths.transpose(0,1) 每行相当于 每个path是属于哪个两个node之间的
        paths_split = paths_weight.mul(paths_over_total)    # 相当于 bw_a_path/bw_path_corresponding_two_node  [218, 1]
        tmp_demand_on_paths = commodities_to_paths.transpose(0,1).matmul(y_true.transpose(0,1)) # 这里真的没写错吗？？？？这个所谓的y_true实际上也不是ytrue把；噢不，这个就是y_true,两个node之间的traffic是固定的，要变动的是split这个flow到别的link上去
        demand_on_paths = tmp_demand_on_paths.mul(paths_split) # bw_a_path_pre * bw_two_nodes_true/bw_two_nodes_pre;
        flow_on_edges = paths_to_edges.transpose(0,1).matmul(demand_on_paths) # 噢噢，这么理解，pre出来的是权重，然后已知了两个node之间的流量带宽，然后根据各个path的权重来分割两个node之间的带宽，最后去计算每个edge的带宽来算MLU
        congestion = flow_on_edges.divide(torch.tensor(np.array([env._capacities])).to(device).transpose(0,1))
        max_cong = torch.max(congestion)
        
        # loss = 1.0 - max_cong if max_cong.item() == 0.0 else max_cong/max_cong.item() # this equation identically equals to 1
        loss = max_cong # ydy: i do not think the above loss func is right;; 现在我觉得，可能是对的了，反正loss就一直在，一直梯度下降就行了
        loss_val = 1.0 if opt == 0.0 else max_cong.item() / opt # no problem, lower is better 
        losses.append(loss)
        loss_vals.append(loss_val)
    
    ret = sum(losses) / len(losses) # ydy: another question. why average?
    ret_val = sum(loss_vals) / len(loss_vals)
    
    return ret, ret_val

# model definition
class NeuralNetworkMaxUtil(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(NeuralNetworkMaxUtil, self).__init__()
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
            nn.Linear(128, output_dim),
            nn.Sigmoid()    # 这里用sigmod是因为，比如输出 a b的所有path上的百分比，因为已知了a到b的demand，就能算每个path上面flow的bw了
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.net(x)
        return logits

def loss_fn_maxutil(y_pred_batch, y_true_batch, env):
    num_nodes = env.get_num_nodes()
    
    losses = []
    loss_vals = []
    batch_size = y_pred_batch.shape[0]
    
    for i in range(batch_size):
        y_pred = y_pred_batch[[i]]
        y_true = y_true_batch[[i]]
        opt = y_true[0][num_nodes * (num_nodes - 1)].item()
        y_true = torch.narrow(y_true, 1, 0, num_nodes * (num_nodes - 1))

    
        y_pred = y_pred + 1e-16 #eps
        paths_weight = torch.transpose(y_pred, 0, 1)
        commodity_total_weight = commodities_to_paths.matmul(paths_weight)  # 这个 [110,218] * [218,1] 就是把每个path上的flow的bw 转为 两两节点之间的 bw
        commodity_total_weight = 1.0 / (commodity_total_weight)
        paths_over_total = commodities_to_paths.transpose(0,1).matmul(commodity_total_weight)   # commodities_to_paths.transpose(0,1) 每行相当于 每个path是属于哪个两个node之间的
        paths_split = paths_weight.mul(paths_over_total)    # 相当于 bw_a_path/bw_path_corresponding_two_node  [218, 1]
        tmp_demand_on_paths = commodities_to_paths.transpose(0,1).matmul(y_true.transpose(0,1)) # 这里真的没写错吗？？？？这个所谓的y_true实际上也不是ytrue把；噢不，这个就是y_true,两个node之间的traffic是固定的，要变动的是split这个flow到别的link上去
        demand_on_paths = tmp_demand_on_paths.mul(paths_split) # bw_a_path_pre * bw_two_nodes_true/bw_two_nodes_pre;
        flow_on_edges = paths_to_edges.transpose(0,1).matmul(demand_on_paths) # 噢噢，这么理解，pre出来的是权重，然后已知了两个node之间的流量带宽，然后根据各个path的权重来分割两个node之间的带宽，最后去计算每个edge的带宽来算MLU
        congestion = flow_on_edges.divide(torch.tensor(np.array([env._capacities])).to(device).transpose(0,1))
        max_cong = torch.max(congestion)
        
        # loss = 1.0 - max_cong if max_cong.item() == 0.0 else max_cong/max_cong.item() # this equation identically equals to 1
        loss = max_cong # ydy: i do not think the above loss func is right;; 现在我觉得，可能是对的了，反正loss就一直在，一直梯度下降就行了
        loss_val = 1.0 if opt == 0.0 else max_cong.item() / opt # no problem, lower is better 
        losses.append(loss)
        loss_vals.append(loss_val)
    
    ret = sum(losses) / len(losses) # ydy: another question. why average?
    ret_val = sum(loss_vals) / len(loss_vals)
    
    return ret, ret_val

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
            nn.ELU(alpha=0.1)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.net(x)
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
        y_true = torch.narrow(y_true, 1, 0, num_nodes * (num_nodes - 1))

        y_pred = y_pred + 0.1 #ELU
        edges_weight = paths_to_edges.transpose(0,1).matmul(torch.transpose(y_pred, 0, 1))
        alpha = torch.max(edges_weight.divide(torch.tensor(np.array([env._capacities])).to(device).transpose(0,1)))
        max_flow_on_tunnel = y_pred / alpha
        max_flow_per_commodity = commodities_to_paths.matmul(max_flow_on_tunnel.transpose(0,1))

        if props.opt_function == "MAXFLOW": #MAX FLOW
            max_mcf = torch.sum(torch.minimum(max_flow_per_commodity.transpose(0,1), y_true))
            
            loss = -max_mcf if max_mcf.item() == 0.0 else -max_mcf/max_mcf.item()
            loss_val = 1.0 if opt == 0.0 else max_mcf.item()/SizeConsts.BPS_TO_GBPS(opt)
        
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
    
    
props = parse_args(sys.argv[1:])
env = history_env.ECMPHistoryEnv(props)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

ctp_coo = env._optimizer._commodities_to_paths.tocoo()
commodities_to_paths = torch.sparse_coo_tensor(np.vstack((ctp_coo.row, ctp_coo.col)), torch.DoubleTensor(ctp_coo.data), torch.Size(ctp_coo.shape)).to(device)
pte_coo = env._optimizer._paths_to_edges.tocoo()
paths_to_edges = torch.sparse_coo_tensor(np.vstack((pte_coo.row, pte_coo.col)), torch.DoubleTensor(pte_coo.data), torch.Size(pte_coo.shape)).to(device)

batch_size = props.so_batch_size
n_epochs = props.so_epochs
concurrent_flow_cdf = None
if props.opt_function == "MAXUTIL":
    NeuralNetwork = NeuralNetworkMaxUtil
    loss_fn = loss_fn_maxutil
elif props.opt_function == "MAXFLOW":
    NeuralNetwork = NeuralNetworkMaxFlowMaxConc
    loss_fn = loss_fn_maxflow_maxconc
elif props.opt_function == "MAXCONC":
    if batch_size == 1:
        batch_size = props.so_max_conc_batch_size
        n_epochs = n_epochs*batch_size
    NeuralNetwork = NeuralNetworkMaxFlowMaxConc
    loss_fn = loss_fn_maxflow_maxconc
    if props.so_mode == SOMode.TEST:
        concurrent_flow_cdf = [0] * (env.get_num_nodes()*(env.get_num_nodes()-1))
else:
    print("Unsupported optimization function. Supported functions: MAXUTIL, MAXFLOW, MAXCOLC")
    assert False

if props.so_mode == SOMode.TRAIN: #train
    # create the dataset
    train_dataset = DmDataset(props, env, False)
    # create a data loader for the train set
    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    #create the model
    model = NeuralNetwork(props.hist_len*env.get_num_nodes()*(env.get_num_nodes()-1), env._optimizer._num_paths) # 输出是 每条<nodei,nodej>的路径上，分配多少带宽
    model.double()
    model.to(device)
    # optimizer
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(n_epochs):
        with tqdm(train_dl) as tepoch:
            epoch_train_loss = []
            loss_sum = loss_count = 0
            for (inputs, targets) in tepoch:        
                # inputs shape: [batch_size, (node_num * (node_num - 1))* hist_len{default 12}]
                # target shape: [batch_size, (node_num * (node_num - 1)) + 1]  concatenate with opt_MLU {no.1's opt -> no. 13, 这一个小时的12个traffic matrix，对应到了下一个小时开始的那个matrix}
                # 
                # put data on the graphics memory   
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
                loss_count += 1
                loss_avg = loss_sum / loss_count
                tepoch.set_postfix(loss=loss_avg)
                # tepoch.set_postfix(loss=loss.cpu().detach().numpy())
        # print('saving...... ' + str(epoch))
        # torch.save(model, 'model_dote_' + str(epoch) + '.pkl')
    #save the model
    # torch.save(model, 'model_dote.pkl')
    torch.save(model, 'model_dote_' + props.ecmp_topo + '.pkl')

elif props.so_mode == SOMode.TEST: #test
    # create the dataset
    test_dataset = DmDataset(props, env, True)
    # create a data loader for the test set
    test_dl = DataLoader(test_dataset, batch_size=1, shuffle=False)
    #load the model
    # model = torch.load('model_dote.pkl').to(device)
    # model = torch.load('model_dote_' + str(n_epochs) + '.pkl').to(device)
    # model = torch.load('model_dote_' + props.ecmp_topo + '.pkl').to(device)
    model = torch.load('model_dote_Abilene-squeeze-links-more1.pkl').to(device)
    model.eval()
    with torch.no_grad():
        with tqdm(test_dl) as tests:
            test_losses = []
            for (inputs, targets) in tests:
                inputs = inputs.to(device)
                targets = targets.to(device)

                pred = model(inputs)
                test_loss, test_loss_val = loss_fn(pred, targets, env)
                test_losses.append(test_loss_val)
            avg_loss = sum(test_losses) / len(test_losses)
            print(f"Test Error: \n Avg loss: {avg_loss:>8f} \n")
            #print statistics to file
            with open(props.graph_base_path + '/' + props.ecmp_topo + '/' + 'so_stats_' + str(n_epochs) + '.txt', 'w') as f:
                import statistics
                dists = [float(v) for v in test_losses]
                dists.sort(reverse=False if props.opt_function == "MAXUTIL" else True)
                f.write('Average: ' + str(statistics.mean(dists)) + '\n')
                f.write('Median: ' + str(dists[int(len(dists) * 0.5)]) + '\n')
                f.write('25TH: ' + str(dists[int(len(dists) * 0.25)]) + '\n')
                f.write('75TH: ' + str(dists[int(len(dists) * 0.75)]) + '\n')
                f.write('90TH: ' + str(dists[int(len(dists) * 0.90)]) + '\n')
                f.write('95TH: ' + str(dists[int(len(dists) * 0.95)]) + '\n')
                f.write('99TH: ' + str(dists[int(len(dists) * 0.99)]) + '\n')
            
            if concurrent_flow_cdf != None:
                concurrent_flow_cdf.sort()
                with open(props.graph_base_path + '/' + props.ecmp_topo + '/' + 'concurrent_flow_cdf.txt', 'w') as f:
                    for v in concurrent_flow_cdf:
                        f.write(str(v / len(dists)) + '\n')

# ydy: not original code
elif props.so_mode == "train_diff_env": #test
    # create the dataset
    train_dataset = DmDataset(props, env, False)
    # create a data loader for the train set
    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    #create the model
    model = torch.load('model_dote_Abilene-squeeze-links-more1.pkl').to(device)
    model.to(device)
    # optimizer
    optimizer = torch.optim.Adam(model.parameters())

    with open("direct_test.txt", "a+") as file:
        for epoch in range(n_epochs):
            with tqdm(train_dl) as tepoch:
                epoch_train_loss = []
                # test_losses = []
                loss_sum = loss_count = 0
                for (inputs, targets) in tepoch:        
                    # inputs shape: [batch_size, (node_num * (node_num - 1))* hist_len{default 12}]
                    # target shape: [batch_size, (node_num * (node_num - 1)) + 1]  concatenate with opt_MLU {no.1's opt -> no. 13, 这一个小时的12个traffic matrix，对应到了下一个小时开始的那个matrix}
                    # 
                    # put data on the graphics memory   
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
                    loss_count += 1
                    loss_avg = loss_sum / loss_count
                    tepoch.set_postfix(loss=loss_avg)
                    
                    file.write(str(loss_avg)+"\n")
                    file.flush()

                    # test_losses.append(loss_val)
                    # avg_loss = sum(test_losses) / len(test_losses)
                    # # print(f"Test Error: \n Avg loss: {avg_loss:>8f} \n")
                    # #print statistics to file
                    # with open('so_stats_' + props.ecmp_topo + '.txt', 'a') as f:
                    #     import statistics
                    #     dists = [float(v) for v in test_losses]
                    #     dists.sort(reverse=False if props.opt_function == "MAXUTIL" else True)
                    #     f.write('Average: ' + str(statistics.mean(dists)) + '\n')
                    #     f.write('Median: ' + str(dists[int(len(dists) * 0.5)]) + '\n')
                    #     f.write('25TH: ' + str(dists[int(len(dists) * 0.25)]) + '\n')
                    #     f.write('75TH: ' + str(dists[int(len(dists) * 0.75)]) + '\n')
                    #     f.write('90TH: ' + str(dists[int(len(dists) * 0.90)]) + '\n')
                    #     f.write('95TH: ' + str(dists[int(len(dists) * 0.95)]) + '\n')
                    #     f.write('99TH: ' + str(dists[int(len(dists) * 0.99)]) + '\n')
                    # tepoch.set_postfix(loss=loss.cpu().detach().numpy())
            # print('saving...... ' + str(epoch))
            # torch.save(model, 'model_dote_' + str(epoch) + '.pkl')
        #save the model
        # torch.save(model, 'model_dote.pkl')
        # torch.save(model, 'model_dote_' + props.ecmp_topo + '.pkl')
    
elif props.so_mode == "train-fixdimen": # train learner with fixed input and output dimension
    NeuralNetwork = AdaptiveNeuralNetworkMaxUtil
    # create the dataset
    train_dataset = DmDataset(props, env, False)
    # create a data loader for the train set
    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    #create the model
    model = NeuralNetwork(props.hist_len*env.get_num_nodes()*(env.get_num_nodes()-1), env._optimizer._num_paths, env.get_num_nodes()) # 输出是 每条<nodei,nodej>的路径上，分配多少带宽
    model.double()
    model.to(device)
    # optimizer
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(n_epochs):
        with tqdm(train_dl) as tepoch:
            epoch_train_loss = []
            loss_sum = loss_count = 0
            for (inputs, targets) in tepoch:        
                # inputs shape: [batch_size, (node_num * (node_num - 1))* hist_len{default 12}]
                # target shape: [batch_size, (node_num * (node_num - 1)) + 1]  concatenate with opt_MLU {no.1's opt -> no. 13, 这一个小时的12个traffic matrix，对应到了下一个小时开始的那个matrix}
                # 
                # put data on the graphics memory   
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
                loss_count += 1
                loss_avg = loss_sum / loss_count
                tepoch.set_postfix(loss=loss_avg)
                # tepoch.set_postfix(loss=loss.cpu().detach().numpy())
        # print('saving...... ' + str(epoch))
        # torch.save(model, 'model_dote_' + str(epoch) + '.pkl')
    #save the model
    # torch.save(model, 'model_dote.pkl')
    torch.save(model, 'fixed_model_' + props.ecmp_topo + '.pkl')
elif props.so_mode == "test-fixdimen": # test adaptive model on the other dataset, especially opensource dataset.
    NeuralNetwork = AdaptiveNeuralNetworkMaxUtil
    # create the dataset
    train_dataset = DmDataset(props, env, False)
    # create a data loader for the train set
    test_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)   
    #load the model
    # model = torch.load('model_dote.pkl').to(device)
    # model = torch.load('model_dote_' + str(n_epochs) + '.pkl').to(device)
    # model = torch.load('meta_model_dote_' + props.ecmp_topo + '.pkl').to(device)
    model = torch.load('meta_model_dote.pkl').to(device) # TODO load 
    # model = torch.load('model_dote_Abilene.pkl').to(device)
    # model = NeuralNetwork(props.hist_len*env.get_num_nodes()*(env.get_num_nodes()-1), env._optimizer._num_paths, env.get_num_nodes())
    # model.load_state_dict('meta_models/model_dote_' + props.ecmp_topo + '.pkl')
    # 替换那两层 nn.linear
    model.input_main_layer = nn.Linear(props.hist_len*env.get_num_nodes()*(env.get_num_nodes()-1), 1320)    # 1320 是我直接写固定了，就把主要模型的维度弄成1320; meta model 那里我也这么做的
    model.net[-2] = nn.Linear(128, env._optimizer._num_paths)
    model.double()
    optimizer = torch.optim.Adam(model.parameters())
    # 这里test模式的时候，只需要在一个我生成的拓扑环境下，就只需要看，需要训几个epoch能达到很低的loss水平
    with open("meta_test_"+props.ecmp_topo+".txt", "a+") as file:
        for epoch in range(n_epochs):
            with tqdm(test_dl) as tests:
                test_losses = []
                loss_count = 0
                for (inputs, targets) in tests:
                    inputs = inputs.to(device)
                    targets = targets.to(device)

                    optimizer.zero_grad()
                    pred = model(inputs)
                    test_loss, test_loss_val = loss_fn(pred, targets, env)
                    test_loss.backward()
                    optimizer.step()
                    test_losses.append(test_loss_val)
                    tests.set_postfix(loss=test_loss_val)
                    
                    loss_count += 1
                    avg_val_loss = sum(test_losses) / loss_count
                    file.write(str(test_loss_val)+"\n")
                    file.flush()

                    avg_loss = sum(test_losses) / len(test_losses)
                    # print(f"Test Error: \n Avg loss: {avg_loss:>8f} \n")
                    # print statistics to file
                with open('so_stats_' + props.ecmp_topo + '.txt', 'a') as f:
                    import statistics
                    dists = [float(v) for v in test_losses]
                    dists.sort(reverse=False if props.opt_function == "MAXUTIL" else True)
                    f.write('Average: ' + str(statistics.mean(dists)) + '\n')
                    f.write('Median: ' + str(dists[int(len(dists) * 0.5)]) + '\n')
                    f.write('25TH: ' + str(dists[int(len(dists) * 0.25)]) + '\n')
                    f.write('75TH: ' + str(dists[int(len(dists) * 0.75)]) + '\n')
                    f.write('90TH: ' + str(dists[int(len(dists) * 0.90)]) + '\n')
                    f.write('95TH: ' + str(dists[int(len(dists) * 0.95)]) + '\n')
                    f.write('99TH: ' + str(dists[int(len(dists) * 0.99)]) + '\n')
            
                if concurrent_flow_cdf != None:
                    concurrent_flow_cdf.sort()
                    with open(props.graph_base_path + '/' + props.ecmp_topo + '/' + 'concurrent_flow_cdf.txt', 'w') as f:
                        for v in concurrent_flow_cdf:
                            f.write(str(v / len(dists)) + '\n')