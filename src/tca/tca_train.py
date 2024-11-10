import os
import sys
from pathlib import Path

folder = Path(__file__).resolve()
sys.path.append(str(folder.parent.parent))

import cupy as cp
import sklearn.metrics
import multiprocessing as mp
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import MinMaxScaler
from env import ABREnv
import ppo2 as network
import torch
import argparse
import tca_rel

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

S_DIM = [6, 8]  # TODO: change fea # TODO original [6,8]
A_DIM = 6
ACTOR_LR_RATE = 1e-4
NUM_AGENTS = 16 # 16
TRAIN_SEQ_LEN = 1000  # take as a train batch
TRAIN_EPOCH = 500000
MODEL_SAVE_INTERVAL = 300
RANDOM_SEED = 42
SUMMARY_DIR = './ppo'
MODEL_DIR = './models'
TRAIN_TRACES = './train/'
TEST_LOG_FOLDER = './test_results/'
LOG_FILE = SUMMARY_DIR + '/log'

# ydy add
TEST_TRACES = ''


NN_MODEL = '/mydata/Pensieve-PPO/src/pretrain/nn_model_ep_80100_bus_my.pth' # None    

class TCA:
    def __init__(self, kernel_type='primal', dim=30, lamb=1, gamma=1):
        self.kernel_type = kernel_type
        self.dim = dim
        self.lamb = lamb
        self.gamma = gamma

    def kernel(self, X1, X2, gamma):
        K = None
        ker = self.kernel_type
        if not ker or ker == 'primal':
            K = X1
        elif ker == 'linear':
            if X2 is not None:
                K = sklearn.metrics.pairwise.linear_kernel(
                    np.asarray(X1).T, np.asarray(X2).T)
            else:
                K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1).T)
        elif ker == 'rbf':
            if X2 is not None:
                K = sklearn.metrics.pairwise.rbf_kernel(
                    np.asarray(X1).T, np.asarray(X2).T, gamma)
            else:
                K = sklearn.metrics.pairwise.rbf_kernel(
                    np.asarray(X1).T, None, gamma)
        return K

    def fit(self, Xs, Xt):
        cp.get_default_memory_pool().free_all_blocks()
        Xs = cp.asarray(Xs)
        Xt = cp.asarray(Xt)
        X = cp.hstack((Xs.T, Xt.T))
        X /= cp.linalg.norm(X, axis=0)
        m, n = X.shape
        ns, nt = len(Xs), len(Xt)
        e = cp.vstack((1 / ns * cp.ones((ns, 1)), -1 / nt * cp.ones((nt, 1))))
        M = e @ e.T
        M = M / cp.linalg.norm(M, 'fro')
        H = cp.eye(n) - 1 / n * cp.ones((n, n))
        print("reach here")
        
        K = self.kernel(cp.asnumpy(X), None, 1)  # Convert CuPy array to NumPy array
        K = cp.asarray(K)  # Convert back to CuPy array
        print("finish kernel")
        
        # Release unused memory
        cp.get_default_memory_pool().free_all_blocks()
        n_eye = m if self.kernel_type == 'primal' else n
        a, b = K @ M @ K.T + self.lamb * cp.eye(n_eye), K @ H @ K.T
        print("ready eigen value")
        
        A_inv_B = cp.linalg.solve(a, b)
        
        # Release unused memory
        cp.get_default_memory_pool().free_all_blocks()
        
        # Convert A_inv_B to NumPy array and perform SVD on CPU
        A_inv_B_cpu = cp.asnumpy(A_inv_B)
        U, S, Vt = np.linalg.svd(A_inv_B_cpu)
        print("solve eigen value")
        
        # Convert the results back to CuPy arrays
        U = cp.asarray(U)
        
        A = U[:, :self.dim]
        Z = A.T @ K
        Z /= cp.linalg.norm(Z, axis=0)

        # Release unused memory
        cp.get_default_memory_pool().free_all_blocks()

        Xs_new, Xt_new = Z[:, :ns].T, Z[:, ns:].T
        Xs_new_np = cp.asnumpy(Xs_new)
        Xt_new_np = cp.asnumpy(Xt_new)
        # 创建 MinMaxScaler 实例，缩放到 [0, 1]
        scaler = MinMaxScaler(feature_range=(0, 1))

        # 进行归一化
        Xs_new_normalized = scaler.fit_transform(Xs_new_np)
        Xt_new_normalized = scaler.fit_transform(Xt_new_np)

        print("done")
        return Xs_new_normalized, Xt_new_normalized
        
def testing(epoch, nn_model, log_file):
    # clean up the test results folder
    os.system('rm -r ' + TEST_LOG_FOLDER)
    #os.system('mkdir ' + TEST_LOG_FOLDER)

    if not os.path.exists(TEST_LOG_FOLDER):
        os.makedirs(TEST_LOG_FOLDER)
    # run test script
    os.system('python /mydata/Pensieve-PPO/src/tca/tca_test.py --nn_model ' + nn_model
              + ' --test_dataset ' + TEST_TRACES
              + ' --log_file ' + TEST_LOG_FOLDER + 'log_sim_ppo')

    # append test performance to the log
    rewards, entropies = [], []
    test_log_files = os.listdir(TEST_LOG_FOLDER)
    for test_log_file in test_log_files:
        reward, entropy = [], []
        with open(TEST_LOG_FOLDER + test_log_file, 'rb') as f:
            for line in f:
                parse = line.split()
                try:
                    entropy.append(float(parse[-2]))
                    reward.append(float(parse[-1]))
                except IndexError:
                    break
        rewards.append(np.mean(reward[1:]))
        entropies.append(np.mean(entropy[1:]))

    rewards = np.array(rewards)

    rewards_min = np.min(rewards)
    rewards_5per = np.percentile(rewards, 5)
    rewards_mean = np.mean(rewards)
    rewards_median = np.percentile(rewards, 50)
    rewards_95per = np.percentile(rewards, 95)
    rewards_max = np.max(rewards)

    log_file.write(str(epoch) + '\t' +
                   str(rewards_min) + '\t' +
                   str(rewards_5per) + '\t' +
                   str(rewards_mean) + '\t' +
                   str(rewards_median) + '\t' +
                   str(rewards_95per) + '\t' +
                   str(rewards_max) + '\n')
    log_file.flush()

    return rewards_mean, np.mean(entropies)
        
def central_agent(net_params_queues, exp_queues):

    assert len(net_params_queues) == NUM_AGENTS
    assert len(exp_queues) == NUM_AGENTS
    
    with open(LOG_FILE + '_test.txt', 'w') as test_log_file:
        actor = network.Network(state_dim=S_DIM, 
                                action_dim=A_DIM,
                                learning_rate=ACTOR_LR_RATE)

        writer = SummaryWriter(SUMMARY_DIR)

        # restore neural net parameters
        nn_model = NN_MODEL
        if nn_model is not None:  # nn_model is the path to file
            actor.load_model(nn_model)
            print('Model restored.')
        
        # while True:  # assemble experiences from agents, compute the gradients
        for epoch in range(TRAIN_EPOCH):
            # synchronize the network parameters of work agent
            actor_net_params = actor.get_network_params()
            for i in range(NUM_AGENTS):
                net_params_queues[i].put(actor_net_params)

            s, a, p, r = [], [], [], []
            for i in range(NUM_AGENTS):
                s_, a_, p_, r_ = exp_queues[i].get()
                s += s_
                a += a_
                p += p_
                r += r_
            s_batch = np.stack(s, axis=0)
            a_batch = np.vstack(a)
            p_batch = np.vstack(p)
            v_batch = np.vstack(r)

            actor.train(s_batch, a_batch, p_batch, v_batch, epoch)
            
            if epoch % MODEL_SAVE_INTERVAL == 0:
                # Save the neural net parameters to disk.
                actor.save_model(SUMMARY_DIR + '/nn_model_ep_' + str(epoch) + '.pth')
                
                avg_reward, avg_entropy = testing(epoch,
                    SUMMARY_DIR + '/nn_model_ep_' + str(epoch) + '.pth', 
                    test_log_file)

                writer.add_scalar('Entropy Weight', actor._entropy_weight, epoch)
                writer.add_scalar('Reward', avg_reward, epoch)
                writer.add_scalar('Entropy', avg_entropy, epoch)
                writer.flush()

# 这个agent没有在训练，只有central agent里面在训练；
# 这个里面的模型虽然叫actor，但是是actor critic，train的时候两个都train，推理的时候只有actor在推理，
def agent(agent_id, net_params_queue, exp_queue):
    env = ABREnv(agent_id)
    actor = network.Network(state_dim=S_DIM, action_dim=A_DIM,
                            learning_rate=ACTOR_LR_RATE)

    # initial synchronization of the network parameters from the coordinator
    actor_net_params = net_params_queue.get()
    actor.set_network_params(actor_net_params)

    for epoch in range(TRAIN_EPOCH):
        obs = env.reset()
        s_batch, a_batch, p_batch, r_batch = [], [], [], []
        for step in range(TRAIN_SEQ_LEN):
            s_batch.append(obs)

            action_prob = actor.predict(
                np.reshape(obs, (1, S_DIM[0], S_DIM[1])))

            # gumbel noise
            noise = np.random.gumbel(size=len(action_prob))
            bit_rate = np.argmax(np.log(action_prob) + noise)

            obs, rew, done, info = env.step(bit_rate)

            action_vec = np.zeros(A_DIM)
            action_vec[bit_rate] = 1
            a_batch.append(action_vec)
            r_batch.append(rew)
            p_batch.append(action_prob)
            if done:
                break
        v_batch = actor.compute_v(s_batch, a_batch, r_batch, done)
        exp_queue.put([s_batch, a_batch, p_batch, v_batch])

        actor_net_params = net_params_queue.get()
        actor.set_network_params(actor_net_params)

def main():
    np.random.seed(RANDOM_SEED)
    torch.set_num_threads(1)
    # inter-process communication queues
    net_params_queues = []
    exp_queues = []
    for i in range(NUM_AGENTS):
        net_params_queues.append(mp.Queue(1))
        exp_queues.append(mp.Queue(1))

    # 思路来说，是一堆actor在推理，然后central_agent收集了他们的探索（每个agent往一个batch里面放1000个点的trace）
    # agent在不断输入 trace，然后产生数据点，central agent在输入那一堆数据点 进行训练
    # create a coordinator and multiple agent processes
    # (note: threading is not desirable due to python GIL)
    coordinator = mp.Process(target=central_agent, args=(net_params_queues, exp_queues))
    coordinator.start()

    agents = []
    for i in range(NUM_AGENTS):
        agents.append(mp.Process(target=agent,
                                 args=(i,
                                       net_params_queues[i],
                                       exp_queues[i])))
    for i in range(NUM_AGENTS):
        agents[i].start()
    # wait unit training is done    # 不能注释掉这个，不然子进程阻塞在那里，但是主进程直接就结束了
    # 并且，coordinate 要在 exp_queues里面放东西，agent才能开始生产数据点，然后coordinate才能训练，所以调试的时候两个都得有
    coordinator.join()

    
def choose_and_combine(src_traces_path, trg_traces_path):
    dataset0_cooked_time, dataset0_cooked_bw, _ = load_trace.load_trace(src_traces_path)
    dataset1_cooked_time, dataset1_cooked_bw, _ = load_trace.load_trace(trg_traces_path)

    # 取最小的length作为最终合成traces的个数
    len_combined = min(len(dataset0_cooked_time),len(dataset1_cooked_time))
    combined_cooked_time = []
    combined_cooked_bw = [] 
    for i in range(len_combined - 1): # 因为我把时刻做差了
        # 计算每两个时刻的差
        dataset0_diffs = np.diff(dataset0_cooked_time[i])
        dataset1_diffs = np.diff(dataset1_cooked_time[i])
        dataset0 = np.column_stack((dataset0_diffs, dataset0_cooked_bw[i][1:]))
        dataset1 = np.column_stack((dataset1_diffs, dataset1_cooked_bw[i][1:]))

        tca_tool = TCA(kernel_type='linear', dim=2, lamb=1, gamma=1) # tca_rel.TCA
        new_dataset0, new_dataset1 = tca_tool.fit(dataset0, dataset1)
        new_dataset = new_dataset1 # np.vstack((new_dataset0, new_dataset1))
        # 再把时间差组装起来
        time_diffs = new_dataset[:, 0]
        cumulative_time = np.cumsum(time_diffs)
        new_dataset[:, 0] = cumulative_time
        # 放缩时间列
        # array_min, array_max = new_dataset[:, 0].min(), new_dataset[:, 0].max()
        # target_min, target_max = np.array(dataset1_cooked_time[i]).min(), np.array(dataset1_cooked_time[i]).max()
        # new_dataset[:, 0] = ((new_dataset[:, 0] - array_min) / (array_max - array_min)) * (target_max - target_min) + target_min
        # 放缩比特率列
        min_val = np.min(dataset1_cooked_bw[i][1:])
        max_val = np.max(dataset1_cooked_bw[i][1:])
        new_dataset[:, 1] = min_val + (new_dataset[:, 1] - new_dataset[:, 1].min()) * (max_val - min_val) / (new_dataset[:, 1].max() - new_dataset[:, 1].min())

        # 将数组保存成指定格式的文件
        combined_save_path = '/mydata/Pensieve-PPO/src/tca/train/combined/distri_0_2/'
        np.savetxt(combined_save_path + "combined_" + str(i), new_dataset, fmt="%.11f", delimiter="\t")

        combined_cooked_time.append(new_dataset[:, 0].tolist())
        combined_cooked_bw.append(new_dataset[:, 1].tolist())
    return combined_cooked_time,combined_cooked_bw
    # traces 貌似没办法选择部分，只能选择提前结束吧
    # portion = 0.0001
    # subset_train_one_X = choose_part_dataset(portion, train_dataset_one)
    # subset_train_two_X = choose_part_dataset(portion, train_dataset_two)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Example script with arguments")
    parser.add_argument('--src', type=str, help='Source Dataset', required=True)
    parser.add_argument('--trg_train', type=str, help='Target Dataset', required=True)
    parser.add_argument('--trg_test', type=str, help='Target Dataset', required=True)
    parser.add_argument('--summary', type=str, help='summary path', required=True)
    parser.add_argument('--test_log_file', type=str, help='Test LOG file path', required=True) # training的log位置无需指定；这个就是 summary+log 的位置
    # parser.add_argument('--log_file', type=str, help='LOG file path', required=True) # training的log位置无需指定；这个就是 summary+log 的位置
    
    # 解析命令行参数
    args = parser.parse_args()
    import load_trace

    # 有了src dataset 和 target dataset 去合成 dataset
    # 选择 这两个distri 里面 length较短的一个，然后每个trace分别进行combine
    # !!!  如果不重新生成数据，要注释这里
    # combined_cooked_time, combined_cooked_bw = choose_and_combine(args.src,args.trg_train)
    # save combined, 然后combined 保存到 `/mydata/Pensieve-PPO/src/tca/train/combined/distri_0_1/`
    load_trace.COOKED_TRACE_FOLDER = '/mydata/Pensieve-PPO/src/tca/train/combined/distri_0_2/'
    # 直接在此处修改 src dataset和target dataset
    SUMMARY_DIR = args.summary
    TEST_TRACES = args.trg_test
    TEST_LOG_FOLDER = args.test_log_file


    # create result directory
    if not os.path.exists(SUMMARY_DIR):
        os.makedirs(SUMMARY_DIR)
    # 调用 main 函数并传递解析后的参数
    main()
