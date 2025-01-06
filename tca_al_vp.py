
import sys
import argparse
import os
import random
import torch
import numpy as np
import datetime
import cv2
from torch.utils.data import Dataset
from config_new import cfg

from torch.optim import AdamW
# from dataset.load_dataset import create_dataset
from models.networking_head import NetworkingHead
from utils.console_logger import ConsoleLogger
from utils.plms_utils import load_plm
from utils.normalize import normalize_data, denormalize_data
from utils.result_notebook import ResultNotebook
from torch.utils.data import DataLoader
from models.pipeline import Pipeline
from models.low_rank import peft_model, print_trainable_parameters

# Set the working directory to the script's directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))
print("Current working directory: ", os.getcwd())

#TODO:Try to use part of the data for baseline and compare in TCA


# These modules are loaded because of TCA and AL
import sys
import time
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import random_split
from scipy.linalg import eigh
from torch import nn
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm
from torch.utils.data import Subset
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
from sklearn.base import BaseEstimator
from scipy.sparse.linalg import eigs
from scipy.spatial.distance import cdist as sp_cdist
from AL_module import *

from scipy.sparse import csr_matrix


print("!!!!!!Script started!!!!!!!")


class TCA:
    def __init__(self, kernel_type='linear', dim=30, lamb=1, gamma=1):
        self.kernel_type = kernel_type
        self.dim = dim
        self.lamb = lamb
        self.gamma = gamma
        self.scaler = MinMaxScaler()
    
    def compute_subset_size(self, total_samples):
        """
        Compute the subset size using the DKW inequality.
        """
        alpha = 50  # Adjusted scaling factor
        delta = 0.01  # Confidence level
        N = total_samples

        # Compute the subset size
        subset_size = int((-N / (2 * (alpha ** 2))) * np.log(delta / 2))

        # Ensure the subset size is positive and doesn't exceed total_samples
        return min(subset_size, total_samples)
    
    def kernel(self, ker, X1, X2=None):
        """
        Compute the kernel matrix based on the specified kernel type.
        """
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
        """
        Perform TCA on source (Xs) and target (Xt) datasets.
        """
        # Combine datasets
        X = np.hstack((Xs.T, Xt.T))
        m, n = X.shape
        ns, nt = len(Xs), len(Xt)

        # Compute total dataset size
        total_samples = ns + nt

        # Compute subset sizes
        subset_size_s = min(self.compute_subset_size(total_samples), ns)
        subset_size_t = min(self.compute_subset_size(total_samples), nt)
        print(f"Using subset size {subset_size_s} for source and {subset_size_t} for target.")


        # Randomly sample subsets
        Xs_subset = Xs[np.random.choice(ns, subset_size_s, replace=False), :]
        Xt_subset = Xt[np.random.choice(nt, subset_size_t, replace=False), :]
        print(f"Xs_subset.shape={Xs_subset.shape}, Xt_subset.shape={Xt_subset.shape}")

        # Combine subsets
        X_subset = np.hstack((Xs_subset.T, Xt_subset.T))
        subset_ns, subset_nt = len(Xs_subset), len(Xt_subset)
        print(f"X_subset.shape={X_subset.shape}")

        # Compute subset-specific M
        e_subset = np.vstack((1 / subset_ns * np.ones((subset_ns, 1)), -1 / subset_nt * np.ones((subset_nt, 1))))
        print(f"e_subset.shape={e_subset.shape}")
        M = e_subset @ e_subset.T
        M = M / np.linalg.norm(M, 'fro')

        # Compute kernel matrix for the subset
        K_subset = self.kernel(self.kernel_type, X_subset.T, X_subset.T)

        # Dimension check
        if K_subset.shape[0] != M.shape[0]:
            raise ValueError(f"Dimension mismatch: K_subset.shape={K_subset.shape}, M.shape={M.shape}")

        # Solve generalized eigenproblem
        n_eye = subset_ns + subset_nt
        a = K_subset @ M @ K_subset.T + self.lamb * np.eye(n_eye)
        b = K_subset @ np.eye(n_eye) - 1 / (n_eye) * np.ones((n_eye, n_eye))
        a = (a + a.T) / 2  # Ensure symmetric
        b = (b + b.T) / 2  # Ensure symmetric

        # epsilon = 1e-6  # Small regularization constant
        # b += epsilon * np.eye(b.shape[0])

        print(f"Dimension of a: {a.shape}, Dimension of b: {b.shape}")

        w, V = scipy.linalg.eig(a, b)
        ind = np.argsort(w)
        self.A = V[:, ind[:self.dim]]

        print(f"Dimension of A: {self.A.shape}")

        # Transform full kernel matrix
        K_full = self.kernel(self.kernel_type, X.T, None)
        print(f"Dimension of K_full: {K_full.shape}")
        print(f"Dimension of A.T: {self.A.T.shape}")

        #TODO: FIX THE DIM MISMATCH HERE!!!
        Z = self.A.T @ K_full
        Z /= np.linalg.norm(Z, axis=0)
        print(f"Dimension of Z: {Z.shape}")

        # Split transformed data back into source and target
        Xs_new, Xt_new = Z[:, :ns].T, Z[:, ns:].T
        print(f"TCA done using subset size {subset_size_s} for source and {subset_size_t} for target.")
        return Xs_new, Xt_new

        

class ViewportDataset(Dataset):
    """
    Wrapper class for viewport dataset.
    """
    def __init__(self, total_traces, total_content_features, videos, users,
                 his_window, fut_window, trim_head, trim_tail, step, for_track=False):
        """
        :param total_traces: total viewport traces
        :param total_content_features: total video content features
        :param videos: video list
        :param users: user list
        :param his_window: historical window
        :param fut_window: future (prediction) window
        :param trim_head: trim some part of the viewport trajectory head
        :param trim_tail: trim some part of the viewport trajectory tail
        :param step: step size of sliding prediction window
        """
        self.total_traces = total_traces
        self.total_content_features = total_content_features
        self.videos = videos
        self.users = users
        self.history_window = his_window
        self.future_window = fut_window
        self.trim_head = trim_head
        self.trim_tail = trim_tail
        self.step = step
        self.for_track = for_track

        # total_traces store the viewport trace of each video and each user
        # we create a list trace_indices to record the indices to the samples in the traces of specific videos and users
        # the idea here is inspired by Quentin Guimard's repo: https://gitlab.com/DVMS_/DVMS
        self.trace_indices = []
        self.content_feature_indices = []  
        for video in videos:
            for user in users:
                trace = self.total_traces[video][user]
                for timestep in range(self.trim_head, len(trace) - self.trim_tail, self.step):
                    self.trace_indices.append((video, user, timestep))
        
        if self.for_track:
            for video in videos:
                image_trace = len(self.total_content_features[video])
                for timestep in range(self.trim_head, image_trace - self.trim_tail, self.step):
                    self.content_feature_indices.append((video, timestep))

    def __len__(self):
        return len(self.trace_indices)

    def __getitem__(self, index):
        """
        With index and self.trace_indices, we can easily access a specific viewport trajectory in the dataset.
        This method is implemented by subclass ViewportDataset360 and ViewportDatasetVV.
        """
        assert self.for_track == True

        if self.for_track:
            video, user, timestep = self.trace_indices[index]
            history = self.total_traces[video][user][timestep - self.history_window:timestep]
            future = self.total_traces[video][user][timestep:timestep + self.future_window]
            history_images = []
            future_images = []
            his_index_start = timestep - self.history_window
            fut_index_start = his_index_end = timestep
            fut_index_end  = self.future_window + fut_index_start
            for c in range(his_index_start,his_index_end):
                history_images.append(self.total_content_features[video][c])
            for c in range(fut_index_start,fut_index_end):
                future_images.append(self.total_content_features[video][c])
            return history, future, history_images, future_images, (video, user, timestep)

        video, user, timestep = self.trace_indices[index]
        history = self.total_traces[video][user][timestep - self.history_window:timestep]
        future = self.total_traces[video][user][timestep:timestep + self.future_window]
        return history, future, (video, user, timestep)
    

def pack_data(dataset_dir, video_user_pairs, frequency, dataset, for_track=False):
    """
    Pack the viewport traces and video content features of corresponding video and user pairs
    into easy-access dict objects
    :param dataset_dir: directory of dataset
    :param video_user_pairs: list of video-user pairs
    :param frequency: the frequency version of the dataset
    :return: total_traces, total_content_features
    """
    pack_traces = {video: {} for video, _ in video_user_pairs}
    for video, user in video_user_pairs:
        data_path = os.path.join(dataset_dir, f'video{video}', f'{frequency}Hz', f'simple_{frequency}Hz_user{user}.csv')
        data = np.loadtxt(data_path, delimiter=',', dtype=np.float32)
        pack_traces[video][user] = data[:, 1:]  # the first column (i.e., column = 0) is timestep, we don't need it
        
    pack_content_features = {video: {} for video, _ in video_user_pairs}
    if for_track:
        image_data_total_path = cfg.dataset_images[dataset]
        for video, user in video_user_pairs:
            image_data_path = os.path.join(image_data_total_path, f'video{video}_images')
            data_path = os.path.join(dataset_dir, f'video{video}', f'{frequency}Hz', f'simple_{frequency}Hz_user{user}.csv')
            tmp_data = np.loadtxt(data_path, delimiter = ',', dtype=np.float32)
            pack_traces[video][user] = tmp_data[:, 1:] 

            if len(pack_content_features[video]) > 0:
                continue
            
            if dataset == 'Jin2022':
                if video in [10, 11, 12, 13, 14, 15, 16, 17, 18]:
                    total_images = 1500
                else:
                    total_images = 1800
            if dataset == 'Wu2017':
                total_images = cfg.Wu2017_video_image[video-1]
            image_freq = int(total_images/len(tmp_data[:, 0]))
            image_names = []
            
            for k in range(1,total_images + 1):
                if ((k-1) % image_freq == 0):
                    image_names.append(os.path.join(image_data_path, f'{k}.png'))
            
            c = 1
            pre_image = None
            for image_name in image_names:
                if os.path.exists(image_name):
                    image = cv2.imread(image_name)
                    image = cv2.resize(image, (224, 224))
                    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
                else:
                    gray_image = pre_image
                pack_content_features[video][c] = gray_image
                c += 1
                pre_image = gray_image
    return pack_traces, pack_content_features


def create_dataset(dataset, dataset_video_split=None, dataset_user_split=None,
                   his_window=cfg.default_history_window, fut_window=cfg.default_future_window,
                   trim_head=cfg.default_trim_head, trim_tail=cfg.default_trim_tail, 
                   frequency=cfg.default_dataset_frequency, step=cfg.default_sample_step,
                   include=('train', 'valid', 'test'), for_track=False):
    """
    Create dataset.
    :param dataset: dataset name
    :param dataset_video_split: train, valid, test split info of videos
    :param dataset_user_split: train, valid, test split info of users
    :param his_window: historical window
    :param fut_window: future (prediction) window
    :param trim_head: trim some part of the viewport trajectory head
    :param trim_tail: trim some part of the viewport trajectory tail
    :param frequency: we have simplify datasets into different frequencies, so we need to specify a frequency to load the coresponding version of dataset
    :param step:the step for sampling viewports
    :param include: inclusion of the splits of dataset
    :return: dataset_train, dataset_valid, dataset_test
    """
    dataset_dir = cfg.dataset[dataset]
    if dataset_video_split is None:
        dataset_video_split = cfg.dataset_video_split[dataset]
    if dataset_user_split is None:
        dataset_user_split = cfg.dataset_user_split[dataset]

    total_video_user_pairs = []
    for split in include:
        videos = dataset_video_split[split]
        users = dataset_user_split[split]
        for video in videos:
            for user in users:
                total_video_user_pairs.append((video, user))
    total_traces, total_content_features = pack_data(dataset_dir, total_video_user_pairs, frequency, dataset, for_track)
    dataset_splits = []
    for split in include:
        dataset_splits.append(

            ViewportDataset(total_traces, total_content_features, dataset_video_split[split],
                            dataset_user_split[split], his_window, fut_window, trim_head, trim_tail, step, for_track)
        )
    return dataset_splits


def apply_tca_on_test(train_dataset, test_dataset):
    # Extract source and target data from total_traces
    Xs_traces = np.vstack([train_dataset.total_traces[video][user] for video in train_dataset.videos for user in train_dataset.users])
    Xt_traces = np.vstack([test_dataset.total_traces[video][user] for video in test_dataset.videos for user in test_dataset.users])
    
    # print("The shape of one trace:", train_dataset.total_traces[train_dataset.videos[2]][train_dataset.users[2]].shape)
    # The shape of one trace: (288, 3)


    print("Shape of Xs_traces:", Xs_traces.shape)
    print("Shape of Xt_traces:", Xt_traces.shape)


    # exit()
    
    # Apply TCA on traces
    tca_traces = TCA(kernel_type='linear', dim=30, lamb=1, gamma=1)
    print("Before TCA on traces: ", Xs_traces.shape, Xt_traces.shape)

    Xs_traces_new, Xt_traces_new = tca_traces.fit(np.asarray(Xs_traces), np.asarray(Xt_traces))

    print("After TCA on traces: ", Xs_traces_new.shape, Xt_traces_new.shape)
    

    #! The transform is necessary?
    Xt_traces_new = ((Xt_traces_new - Xt_traces_new.min()) / (Xt_traces_new.max() - Xt_traces_new.min())) * (Xt_traces.max() - Xt_traces.min()) + Xt_traces.min() # transform

    # Update test dataset with transformed traces
    index = 0
    for video in test_dataset.videos:
        for user in test_dataset.users:
            length = len(test_dataset.total_traces[video][user])
            # Assign each row back to the corresponding video and user
            test_dataset.total_traces[video][user] = Xt_traces_new[index:index + length]
            index += length

    #! This is for features. Notice that we do not need the return value since we have already saved the file in a new path
    
    Xs_features_new, Xt_features_new = process_and_apply_tca()
    
    return train_dataset, test_dataset


def process_and_apply_tca():
    # Extract the train and valid indices for Jin2022
    train_indices = cfg.dataset_video_split['Jin2022']['train']
    valid_indices = cfg.dataset_video_split['Jin2022']['valid']

    print("Train indices:", train_indices)
    print("Valid indices:", valid_indices)

    # Calculate the number of target indices (2/3 of valid indices)
    num_target_indices = len(valid_indices) * 2 // 3

    # Extract the target indices
    target_indices = valid_indices[:num_target_indices]

    print("Target indices:", target_indices)

    # Define the base path for the image features
    base_path = '/projects/bcrn/yliang7/research/NetLLM/viewport_prediction/data/images/Jin2022images/features'

    # Get the paths for source and target folders
    source_folders = [os.path.join(base_path, f'video{index}_images') for index in train_indices]
    target_folders = [os.path.join(base_path, f'video{index}_images') for index in target_indices]

    print("Source folders:", source_folders)
    print("Target folders:", target_folders)

    def read_and_stack_features(folders):
        all_features = []
        file_paths = []
        for folder in folders:
            for file_name in os.listdir(folder):
                if file_name.endswith('.pth'):
                    file_path = os.path.join(folder, file_name)
                    loaded_tensor_dict = torch.load(file_path)
                    for key, value in loaded_tensor_dict.items():
                        all_features.append(value.detach().numpy().flatten())
                    file_paths.append(file_path)
        return np.vstack(all_features), file_paths

    # Read and stack features for source and target
    Xs_features, source_file_paths = read_and_stack_features(source_folders)
    Xt_features, target_file_paths = read_and_stack_features(target_folders)

    print("Shape of Xs_features:", Xs_features.shape)
    print("Shape of Xt_features:", Xt_features.shape)

    # Apply TCA on traces
    tca_traces = TCA(kernel_type='linear', dim=30, lamb=1, gamma=1)
    print("Before TCA on traces: ", Xs_features.shape, Xt_features.shape)
    print("Before TCA on features: ", Xs_features.shape, Xt_features.shape)
    Xs_features_new, Xt_features_new = tca_traces.fit(np.asarray(Xs_features), np.asarray(Xt_features))

    print("After TCA on traces: ", Xs_features_new.shape, Xt_features_new.shape)

    # Transform Xt_features_new
    Xt_features_new = ((Xt_features_new - Xt_features_new.min()) / (Xt_features_new.max() - Xt_features_new.min())) * (Xt_features.max() - Xt_features.min()) + Xt_features.min()

    print("Transformed Xt_features_new:", Xt_features_new.shape)

    # Save the transformed features back to .pth files
    def save_transformed_features(folders, transformed_features, file_paths, suffix='_tca'):
        feature_index = 0
        for folder, file_path in zip(folders, file_paths):
            new_folder = folder + suffix
            os.makedirs(new_folder, exist_ok=True)
            new_file_path = os.path.join(new_folder, os.path.basename(file_path))
            loaded_tensor_dict = torch.load(file_path)
            for key in loaded_tensor_dict.keys():
                loaded_tensor_dict[key] = torch.tensor(transformed_features[feature_index]).view(1, -1)
                feature_index += 1
            torch.save(loaded_tensor_dict, new_file_path)

    save_transformed_features(source_folders, Xs_features_new, source_file_paths)
    save_transformed_features(target_folders, Xt_features_new, target_file_paths)

    return Xs_features_new, Xt_features_new

def save_model(args, model, save_dir):
    """
    save fune-tune model
    """
    if args.rank != -1:
        # save low rank matrices
        model.plm.save_pretrained(save_dir)
        # save other modules except plm
        torch.save(model.modules_except_plm.state_dict(), os.path.join(save_dir, 'modules_except_plm.bin'))
    else:
        # low rank matrices are disabled, save whole model
        torch.save(model.state_dict(), os.path.join(save_dir, 'model.bin'))


def load_model(args, model, model_dir):
    """
    load fune-tune model

    :return: the pretrained model corresponding to using model_dir
    """
    if args.rank != -1:
        # load low rank matrices
        model.plm.load_adapter(model_dir, adapter_name='default')
        # load other modules except plm
        model.modules_except_plm.load_state_dict(torch.load(os.path.join(model_dir, 'modules_except_plm.bin')))
    else:
        # low rank matrices are disabled, load whole model
        model.load_state_dict(torch.load(os.path.join(model_dir, 'model.bin')))
    return model

def adapt(args, pipeline, train_dl, test_train_dl, test_test_dl, models_dir, grad_accum_steps):
    file_prefix = f'his_{args.his_window}_fut_{args.fut_window}_ss_{args.sample_step}_epochs_{args.epochs}_bs_{args.bs * args.grad_accum_steps}_'\
                  f'lr_{args.lr}_seed_{args.seed}_rank_{args.rank}_scheduled_sampling_{args.scheduled_sampling}_task_name_{args.task_name}'
    checkpoint_path = os.path.join(models_dir, file_prefix, 'checkpoint')
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    best_model_path = os.path.join(models_dir, file_prefix, 'best_model')
    if not os.path.exists(best_model_path):
        os.makedirs(best_model_path)
    console_log = open(os.path.join(models_dir, file_prefix + '_console.log'), 'w')
    sys.stdout = ConsoleLogger(sys.__stdout__, console_log)

    #! Remember this!
    if args.resume:
        pipeline = load_model(args, pipeline, args.resume_path)
        print('Resume weights for training from:', args.resume_path)

    if not args.freeze_plm:
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in pipeline.plm.named_parameters() if not any(nd in n for nd in no_decay)], 
            'weight_decay': args.weight_decay, 'lr': args.lr},
            {'params': [p for n, p in pipeline.plm.named_parameters() if any(nd in n for nd in no_decay)], 
            'weight_decay': 0.0, 'lr': args.lr},
            {'params': pipeline.embed_vp.parameters(), 'weight_decay': args.weight_decay, 'lr': args.lr},
            {'params': pipeline.embed_ln.parameters(), 'weight_decay': args.weight_decay, 'lr': args.lr},
            {'params': pipeline.embed_multimodal.parameters(), 'weight_decay': args.weight_decay, 'lr': args.lr}
        ]
        optimizer = AdamW(optimizer_grouped_parameters)
    else:
        # only tune networking head and multimodal encoder
        optimizer_grouped_parameters = [
            {'params': pipeline.embed_vp.parameters(), 'weight_decay': args.weight_decay, 'lr': args.lr},
            {'params': pipeline.embed_ln.parameters(), 'weight_decay': args.weight_decay, 'lr': args.lr},
            {'params': pipeline.embed_multimodal.parameters(), 'weight_decay': args.weight_decay, 'lr': args.lr}
        ]
        optimizer = AdamW(optimizer_grouped_parameters)

    assert args.epochs_per_valid is None or args.steps_per_valid is None, "You can only specify args.epochs_per_valid or args.steps_per_valid."

    global_step = 0
    report_loss_per_steps = args.report_loss_per_steps
    tot_loss = 0
    log_loss = 0
    best_loss = float('inf')
    best_epoch, best_step = 0, 0

    #! Following AL
    ##initialize 10% dataset as initialize dataset
    i = 0
    ##AL start initialize dataset
    initialize = True
    active_indices,pool_indices,simulation_indices,remaining_indices,initialize,selection_cost = AL_Select(test_train_dl,initialize,i)
    start_time = time.time()
    cost_utility = 0

    print(f"selection_cost: {selection_cost}")
    subset_sampler = SubsetRandomSampler(remaining_indices)
    subset_dataloader = DataLoader(test_train_dl.dataset, batch_size=args.bs, sampler=subset_sampler)
    #smooth
    prev_smoothed_delta_cost_ratio = None
    prev_smoothed_diff_sum_cost_ratio = None
    smooth_delta_cost_ratios = []  # List to store (delta / iter_training_cost) values
    smooth_diff_sum_cost_ratios = []
    window_size = 1
    #initialzie value
    iter_training_cost = 0
    current_uncertainty = 0
    last_uncertainty = 100
    #simulation choice
    sample_size = len(simulation_indices)  # Fixed sample size of 100
    dataset_size = len(test_train_dl.dataset)
    apply_check = False
    simulation_epoch = 0
    
    number = 0

        
    print(f'Training on {args.train_dataset} - bs: {args.bs} - lr: {args.lr} - seed: {args.seed}')
    
    for epoch in range(args.epochs):
        pipeline.train()
        assert args.using_multimodal == True
        for step, (history, future, history_images, future_images, video_user_info) in enumerate(train_dl):  
            # print(f'Step: {step} out of {len(train_dl)}', flush=True)

            #! Print the total number of steps
            # print(f'Total steps: {len(dataloader_train)}', flush=True)
            # total steps: 8820

            global_step += 1
            history, future = history.to(args.device), future.to(args.device)
            history = normalize_data(history, args.train_dataset)
            future = normalize_data(future, args.train_dataset)
            # using scheduled sampling
            if args.scheduled_sampling:
                if np.random.rand() > args.mix_rate:
                    loss = pipeline(history, future, video_user_info, teacher_forcing=True)
                else:
                    loss = pipeline(history, future, video_user_info, teacher_forcing=False)
            else:
                loss = pipeline(history, future, video_user_info, teacher_forcing=True)
            tot_loss += loss.item()
            loss = loss / grad_accum_steps
            loss.backward()
            torch.nn.utils.clip_grad_norm_(pipeline.plm.parameters(), 1.0)

            # perform gradient accumulation update
            if ((step + 1) % grad_accum_steps == 0) or (step + 1 == len(train_dl)):
                optimizer.step()
                optimizer.zero_grad()
            
            # report training loss
            if global_step % report_loss_per_steps == 0:
                print("Epoch {}, global_step {}, average loss: {}".format(epoch, global_step, (tot_loss - log_loss) / report_loss_per_steps), flush=True)
                log_loss = tot_loss
            
            # for debug
            # if global_step >= 100:
            #     save_model(args, pipeline, best_model_path)
            #     break
            
        
        # save checkpoint by save_checkpoint_per_epoch
        if args.save_checkpoint_per_epoch is not None and epoch % args.save_checkpoint_per_epoch == 0 and epoch > 0:
            save_checkpoint_path = os.path.join(checkpoint_path, f'epoch{epoch}') # save checkpoint
            if not os.path.exists(save_checkpoint_path):
                os.makedirs(save_checkpoint_path)
            save_model(args, pipeline, save_checkpoint_path)
            print('save checkpoint at', save_checkpoint_path)

    print('Done adaptation, average training loss =', tot_loss / global_step)


    global_step_for_ft = 0

    #! Continue training on test_train dataset for 2 epochs, evaluating on test_test every 1 epoch (hyperparameter to be determined)
    additional_epoch = 10
    for epoch in range(additional_epoch):
        if apply_check == True:
            apply_check = False
            simulation_epoch = 0
            train_loss = []
            train_mape = []
            val_loss = []
            val_mape = []
            best_mape = 100
            train_loss_per_epoch = []
            train_mape_per_epoch = []
            prev_smoothed_delta_cost_ratio = None
            prev_smoothed_diff_sum_cost_ratio = None
            smooth_delta_cost_ratios = []  # List to store (delta / iter_training_cost) values
            smooth_diff_sum_cost_ratios = []
            window_size = 1
            #initialzie value
            iter_training_cost = 0
            current_uncertainty = 0
            last_uncertainty = 100
        
        #time to calculation
        epoch_time_start = time.time()
        #save each epoch uncertainties
        uncertainties = []

        pipeline.train()
        assert args.using_multimodal == True
        for step, (history, future, history_images, future_images, video_user_info) in enumerate(test_train_dl): 

            # print(f'Step: {step} out of {len(test_train_dl)}', flush=True)


            global_step_for_ft += 1
            history, future = history.to(args.device), future.to(args.device)
            history = normalize_data(history, args.train_dataset)
            future = normalize_data(future, args.train_dataset)
            # using scheduled sampling
            if args.scheduled_sampling:
                if np.random.rand() > args.mix_rate:
                    loss = pipeline(history, future, video_user_info, teacher_forcing=True)
                else:
                    loss = pipeline(history, future, video_user_info, teacher_forcing=False)
            else:
                loss = pipeline(history, future, video_user_info, teacher_forcing=True)
            tot_loss += loss.item()

            #TODO: Determine the place to this put this snippet later
            uncertainties.append(loss.item())

            loss = loss / grad_accum_steps
            loss.backward()
            torch.nn.utils.clip_grad_norm_(pipeline.plm.parameters(), 1.0)

            # perform gradient accumulation update
            if ((step + 1) % grad_accum_steps == 0) or (step + 1 == len(train_dl)):
                optimizer.step()
                optimizer.zero_grad()
            
            # report training loss
            if global_step_for_ft % report_loss_per_steps == 0:
                print("Epoch {}, global_step_for_ft {}, average loss: {}".format(epoch, global_step_for_ft, (tot_loss - log_loss) / report_loss_per_steps), flush=True)
                log_loss = tot_loss

            # for debug propose
            if global_step_for_ft >= 200:
                save_model(args, pipeline, best_model_path)
                break

        ##parameter of uncertainty and time cost

        current_uncertainty = sum(uncertainties)
        print("uncertainties shape:", len(uncertainties))
        epoch_time_end = time.time()  

        #TODO: Scale this time cost later
        iter_training_cost = get_time_cost(iter_training_cost,epoch_time_end,epoch_time_start)
        print('iter_training_cost {}'.format(iter_training_cost))

        ##simulation of active learning
        #TODO: uncertainty() function inside uncertainties_simulation() is what we should modify!


        #! We can now reach here
        data_uncertainty = uncertainties_simulation(test_train_dl,dataset_size,sample_size,iter_training_cost,simulation_indices, pipeline, len(active_indices),
        len(pool_indices))

        ##metrics to stop al
        #! In this part, the para to pass in are all related to AL itself
        result, prev_smoothed_delta_cost_ratio, prev_smoothed_diff_sum_cost_ratio = calculate_and_compare_metrics(
            uncertainties,
            last_uncertainty,
            current_uncertainty,
            dataset_size,
            iter_training_cost,
            data_uncertainty,
            prev_smoothed_delta_cost_ratio,
            prev_smoothed_diff_sum_cost_ratio,
            smooth_delta_cost_ratios,
            smooth_diff_sum_cost_ratios,
            window_size,
            simulation_epoch,
            len(active_indices),
            len(pool_indices),
            selection_cost
        )
        simulation_epoch+=1

        # For debugging
        # result = True

        if result == False:
            apply_check = True
            if len(pool_indices) <= 0:
                break
            
            #TODO: active_learning_iteration is what we should modify!
            number,new_indices,cost_utility = active_learning_iteration(cost_utility,i, pipeline, test_train_dl, pool_indices, args.bs)
            i+=1
            active_indices.extend(new_indices)
            pool_indices = list(set(pool_indices) - set(new_indices))
            selection_cost = cost_utility
            simulation_size = min(max(50,int(0.1 * len(active_indices))),1000)
            simulation_indices = random.sample(new_indices, simulation_size)
            remaining_indices = list(set(active_indices) - set(simulation_indices))
            subset_sampler = SubsetRandomSampler(remaining_indices)
            subset_dataloader = DataLoader(test_train_dl.dataset, batch_size=args.bs, sampler=subset_sampler)
        else:
            last_uncertainty = current_uncertainty

        # Every 5 epochs, test the model on test_test
        if (epoch + 1) % 1 == 0:
            test_model_on_test_test(pipeline, test_test_dl, checkpoint_path)
        

#! This function varies from model to model
def uncertainty(pipeline, dataloader):
    pipeline.eval()
    with torch.no_grad():

        uncertainties = []

        assert args.using_multimodal == True

        for step, (history, future, history_images, future_images, video_user_info) in enumerate(dataloader):

            history, future = history.to(args.device), future.to(args.device)
            history = normalize_data(history, args.train_dataset)
            future = normalize_data(future, args.train_dataset)
            loss = pipeline(history, future, video_user_info, teacher_forcing=False)
            uncertainties.append(loss.item())
        return uncertainties
    


def uncertainties_simulation(tr_dataloader,dataset_size,sample_size,iter_training_cost,simulation_indices, pipeline, AL_select, AL_leftover):
    #TODO1: Finish this function
    top_k = AL_select / AL_leftover

    sample_dataloader = DataLoader(
        tr_dataloader.dataset,  # Use the dataset attribute
        batch_size=tr_dataloader.batch_size,
        sampler=SubsetRandomSampler(simulation_indices)  # Use a sampler instead of Subset
    )

    sample_scale = dataset_size / sample_size
    budget = iter_training_cost / sample_scale

    #? Not sure?
    ### modified based on model
    sampled_costs = []
    for _ in simulation_indices:
        #! In VP task, the time cost is 1
        sampled_costs.append(1)
        
    sampled_uncertainties = []  
  
    sampled_uncertainties = uncertainty(pipeline, sample_dataloader)
    data_uncertainty = AL_intrain(sampled_uncertainties,budget,sampled_costs,top_k)
    return data_uncertainty

def uncertainty_select(pipeline, dataloader):
    pipeline.eval()
    uncertainties = []
    with torch.no_grad():
        
        for history, future, history_images, future_images, video_user_info in dataloader:

            inputs = history.to(args.device)
            confidence_level = 0.95
            t_value = t.ppf((1 + confidence_level) / 2, len(inputs) - 1)
            intervals = []

            for i in range(inputs.shape[0]):  

                #! Modified
                i_tensor = torch.tensor([i], dtype=torch.float32, device=inputs.device)

                # Loop through each sample in the batch
                # Prediction interval for latency
                interval = t_value * torch.sqrt(
                    1 + (1 / len(inputs)) + 
                    ((i_tensor - inputs.mean(dim=0))**2).sum() / ((inputs - inputs.mean(dim=0))**2).sum()
                )
                intervals.append(interval.cpu().numpy())

            uncertainties.extend(intervals)

        return uncertainties

def active_learning_iteration(cost_utility,i,pipeline, test_train_dl, pool_indices, batch_size):
    print(f"Starting active learning iteration {i}", flush=True)
    
    pool_loader = DataLoader(Subset(test_train_dl.dataset, pool_indices), batch_size=batch_size, shuffle=False)
    
    print("Calculating uncertainty values...", flush=True)
    uncertainty_values = uncertainty_select(pipeline, pool_loader)
   
    print("Calculating costs...", flush=True)
    # Create a dictionary mapping each pool index to its corresponding cost

    #! In VP task, the time cost is 1
    costs = {1 for _ in pool_indices}
    
    print(f"pool_indices: {len(pool_indices)}", flush=True)
    print(f"Costs: {len(costs)}", flush=True)
    print(f"uncertainty_values: {len(uncertainty_values)}", flush=True)
    print("Sample of Uncertainty values:", flush=True)
    print("First 10:", uncertainty_values[:10], flush=True)
    
    assert len(uncertainty_values) == len(costs), "Mismatch between number of uncertainty values and costs"
    
    print("Calculating uncertainty-cost ratios...", flush=True)
    # Compute uncertainty-cost ratios using the dictionary
    uncertainty_cost_ratios = [u / costs[l] for u, l in zip(uncertainty_values, pool_indices)]
    
    num_to_select = int(0.1 * len(test_train_dl.dataset))
    uncertainty_weights = [1/u for u in uncertainty_cost_ratios]
    
    selected_indices = []
    current_cost = 0
    
    if i == 0:
        print("First iteration (i == 0)", flush=True)
        print("Selecting indices based on number to select...", flush=True)
        while len(selected_indices) < num_to_select:
            selected_data = random.choices(pool_indices, weights=uncertainty_weights, k=1)[0]
            if selected_data not in selected_indices:  # Avoid duplicates
                selected_indices.append(selected_data)
                selected_cost = costs[selected_data]
                current_cost += selected_cost
                # print(f"selected_data: {selected_data}, selected_cost: {selected_cost}, cost_utility: {current_cost}", flush=True)
    else:
        print(f"Iteration {i}", flush=True)
        print("Selecting indices based on cost utility...", flush=True)
        while current_cost < cost_utility:
            selected_data = random.choices(pool_indices, weights=uncertainty_weights, k=1)[0]
            if selected_data not in selected_indices:  # Avoid duplicates
                selected_indices.append(selected_data)
                selected_cost = costs[selected_data]
                current_cost += selected_cost
                # print(f"selected_data: {selected_data}, selected_cost: {selected_cost}, cost_utility: {current_cost}", flush=True)
    
    cost_utility = current_cost
    print(f"Final cost_utility: {cost_utility}", flush=True)
    print(f"Number of selected points: {len(selected_indices)}", flush=True)
    
    return num_to_select, selected_indices, cost_utility

def test_model_on_test_test(pipeline, test_test_dl, checkpoint_path):
    pipeline.eval()
    with torch.no_grad():
        validata_checkpoint_path = os.path.join(checkpoint_path)
        if not os.path.exists(validata_checkpoint_path):
            os.makedirs(validata_checkpoint_path)

        save_model(args, pipeline, validata_checkpoint_path)
        print(f'Checkpoint saved at', checkpoint_path)
        valid_loss = []
        for index, (history, future, history_images, future_images, video_user_info) in enumerate(test_test_dl):
            history, future = history.to(args.device), future.to(args.device)
            history = normalize_data(history, args.train_dataset)
            future = normalize_data(future, args.train_dataset)
            loss = pipeline(history, future, video_user_info, teacher_forcing=False)
            valid_loss.append(loss.item())
        valid_loss = sum(valid_loss) / len(valid_loss)
        print(f'Valid loss: {valid_loss}')
        pipeline.train()
        return valid_loss

def run(args):
    assert args.train_dataset in cfg.dataset_list 
    assert args.test_dataset in cfg.dataset_list
    assert args.plm_type in cfg.plm_types
    assert args.plm_size in cfg.plm_sizes
    assert args.trim_head >= args.his_window and args.trim_tail >= args.fut_window

    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    if args.rank != -1:
        models_dir = os.path.join(cfg.plms_finetuned_dir, f'{args.plm_type}_{args.plm_size}_low_rank', 
                              f'freeze_plm_{args.freeze_plm}', args.train_dataset, f'{args.dataset_frequency}Hz')
        results_dir = os.path.join(cfg.results_dir, f'{args.plm_type}_{args.plm_size}_low_rank', 
                               f'freeze_plm_{args.freeze_plm}', args.test_dataset, f'{args.dataset_frequency}Hz')
    else:
        models_dir = os.path.join(cfg.plms_finetuned_dir, f'{args.plm_type}_{args.plm_size}', 
                              f'freeze_plm_{args.freeze_plm}', args.train_dataset, f'{args.dataset_frequency}Hz')
        results_dir = os.path.join(cfg.results_dir, f'{args.plm_type}_{args.plm_size}', 
                               f'freeze_plm_{args.freeze_plm}', args.test_dataset, f'{args.dataset_frequency}Hz')
    if not os.path.exists(models_dir): 
        os.makedirs(models_dir)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # plm, tokenizer, _ = load_plm(args.plm_type, os.path.join(cfg.plms_dir, args.plm_type, args.plm_size), plm_size=args.plm_size, 
    #                                  device_input_side=args.device, device_output_side=args.device_out, device_middle_side=args.device_mid)
    # if (args.plm_type == 'opt' or args.plm_type == 'gpt2') and args.plm_size!= 'large':  # other plm can simply be loaded on one device
    #     plm = plm.to(args.device)
    
    # if args.rank != -1:
    #     plm = peft_model(plm, args.plm_type, args.rank)
        
    # # set up networking head
    # input_dim = plm.hidden_size
    # out_dim = 3  # = the number of viewport coordinates
    # if args.plm_type == 'opt' and args.plm_size == 'xxs':
    #     networking_head = NetworkingHead(input_dim=512, output_dim=out_dim, fut_window=args.fut_window).to(args.device_out)
    # else:
    #     networking_head = NetworkingHead(input_dim=input_dim, output_dim=out_dim, fut_window=args.fut_window).to(args.device_out)
    # plm.set_networking_head(networking_head)
    # print('PLM model architecture:')
    # print(plm)
    
    # if args.plm_type == 'llama':
    #     embed_size = 4096

    # pipeline = Pipeline(plm, fut_window=args.fut_window, device=args.device, embed_size=embed_size, frequency=args.dataset_frequency, using_multimodal=args.using_multimodal, dataset=args.train_dataset)

    if args.compile:
        assert torch.__version__ >= '2.0.0', 'Compile model requires torch version >= 2.0.0, but current torch version is ' + torch.__version__
        print("\033[33mWarning:\033[0m There seems to be some bugs in torch.compile. If batch size is too large, it will raise errors (I don't know why this happens).")
        prompt_model = torch.compile(prompt_model).to(args.device)  # recommend to compile model when you are using PyTorch 2.0
    
    torch.set_float32_matmul_precision('high')

    if args.adapt:
        #! Missing for_track=args.using_multimodal in the original repo
        raw_dataset_train, raw_dataset_valid = create_dataset(args.train_dataset, his_window=args.his_window, 
                                                              fut_window=args.fut_window, trim_head=args.trim_head, trim_tail=args.trim_tail,
                                                              include=['train', 'valid'], frequency=args.dataset_frequency, step=args.sample_step, for_track=args.using_multimodal)
        raw_dataset_test = create_dataset(args.test_dataset, his_window=args.his_window, fut_window=args.fut_window,
                                          trim_head=args.trim_head, trim_tail=args.trim_tail, include=['test'], frequency=args.dataset_frequency, step=args.sample_step)[0]
        
        dataloader_test = DataLoader(raw_dataset_test, batch_size=args.bs, shuffle=True, pin_memory=True)
        
        # print the shape of the dataset, not the content
        print("##########")
        print('raw_dataset_train:', raw_dataset_train, "len(raw_dataset_train):", len(raw_dataset_train), "len(raw_dataset_train[0]):", len(raw_dataset_train[0]))
        print("##########")

        print("Apply TCA on the domain dataset...")
        
        #! This is the TCA on the training dataset
        #! This is currently commented out

        #!For debug
        # print("Skip TCA")
        # The raw_dataset_finetune is used for further training
        raw_dataset_train, raw_dataset_finetune = apply_tca_on_test(raw_dataset_train, raw_dataset_valid)
        raw_dataset_train, raw_dataset_finetune = raw_dataset_train, raw_dataset_valid

        print("##########")
        print('raw_dataset_train:', raw_dataset_train, "len(raw_dataset_train):", len(raw_dataset_train), "len(raw_dataset_train[0]):", len(raw_dataset_train[0]))
        print("##########")

        # exit the program
        # exit(0)
        
        #! Until now, we have the raw_dataset_train and raw_dataset_finetune, both of which are ViewportDataset, which is inherited from the Dataset class

        #! Following AL
        portion = 1
        subset_train_X = raw_dataset_train
        subset_test_X = raw_dataset_finetune

        dataset_size = len(subset_test_X)
        #! The sequence is maintained, so we can make sure that 2/3 of the dataset is the test_train dataset that we define in the NetLLM Config file

        train_size = int(2 / 3 * dataset_size)  # x% 作为训练集

        # 非随机顺序分割ç
        train_indices = list(range(train_size))
        test_indices = list(range(train_size, dataset_size))

        subset_test_train_X = Subset(subset_test_X, train_indices)
        subset_test_test_X = Subset(subset_test_X, test_indices)

        print(f"Number of samples in subset_train_X: {len(subset_train_X)}")

        train_dl = DataLoader(subset_train_X, batch_size=args.bs, shuffle=True, pin_memory=True)
        test_train_dl = DataLoader(subset_test_train_X, batch_size=args.bs, shuffle=True, pin_memory=True)
        test_test_dl = DataLoader(subset_test_test_X, batch_size=args.bs, shuffle=True, pin_memory=True)

        # dataloader_train = DataLoader(raw_dataset_train, batch_size=args.bs, shuffle=True, pin_memory=True)
        # dataloader_valid = DataLoader(raw_dataset_valid, batch_size=args.bs, shuffle=False, pin_memory=True)
        # adapt(args, pipeline, dataloader_train, dataloader_valid, models_dir, args.grad_accum_steps)
        adapt(args, pipeline, train_dl, test_train_dl, test_test_dl, models_dir, args.grad_accum_steps)

        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process the input parameters to train the network.')
    
    # ========== model/plm settings related arguments ==========
    parser.add_argument('--task_name', action="store", dest='task_name', type=str, help='the task name to run the experiment.')
    parser.add_argument('--adapt', action="store_true", help='adapt llm.')
    parser.add_argument('--test', action="store_true", help='test llm.')
    parser.add_argument('--plm-type', action="store", dest='plm_type', help='type of plm.', default='t5-lm')
    parser.add_argument('--plm-size', action="store", dest='plm_size', help='size of plm.', default='base')
    parser.add_argument('--model-path', action="store", dest='model_path', type=str, help='(Optional) The directory of model weights to be loaded for testing.')
    parser.add_argument('--device', action='store', dest='device', help='the device (cuda or cpu) to run experiment.')
    parser.add_argument('--device-out', action='store', dest='device_out', help='the device (cuda or cpu) to place the split of model near the output.')
    parser.add_argument('--device-mid', action='store', dest='device_mid', help='the device (cuda or cpu) to place the split of model between the input and output.')
    parser.add_argument('--freeze-plm', action='store_true', dest='freeze_plm', help='freeze weights of plm during training')
    parser.add_argument('--compile', action='store_true', dest='compile', help='(Optional) Compile model for speed up (available only for PyTorch 2.0).')
    parser.add_argument('--resume', action='store_true', dest='resume', help='(Optional) Resume model weights from checkpoint for training.')
    
    # ========== dataset settings related arguments ==========
    parser.add_argument('--train-dataset', action='store', dest='train_dataset', help='Dataset for training.')
    parser.add_argument('--test-dataset', action='store', dest='test_dataset', help='Dataset for testing.')

    # ========== dataset loading/processing settings related arguments ==========
    parser.add_argument('--his-window', action='store', dest='his_window',
                        help='(Optional) Historical window (default 10)', type=int)
    parser.add_argument('--fut-window', action='store', dest='fut_window',
                        help='(Optional) Future (prediction) window (default 10).', type=int)
    parser.add_argument('--trim-head', action='store', dest='trim_head',
                        help='(Optional) Trim some part of the viewport trajectory head (default 30).', type=int)
    parser.add_argument('--trim-tail', action='store', dest='trim_tail',
                        help='(Optional) Trim some part of the viewport trajectory tail (default 30).', type=int)
    parser.add_argument('--dataset-frequency', action='store', dest='dataset_frequency',
                        help='(Optional) The frequency version of the dataset (default 10).', type=int)
    parser.add_argument('--sample-step', action='store', dest='sample_step',
                        help='(Optional) The steps for sampling viewports (default 1).', type=int)

    # ========== training related settings ==========
    parser.add_argument('--epochs', action="store", dest='epochs', help='(Optional) Neural network learning epochs.', type=int)
    parser.add_argument('--epochs-per-valid', action='store', dest='epochs_per_valid', type=int,
                        help='(Optional) The number of epochs per validation (default 3).')
    parser.add_argument('--steps-per-valid', action='store', dest='steps_per_valid', type=int,
                        help='(Optional) The number of steps per validation (default 50).')
    parser.add_argument('--report-loss-per-steps', action='store', dest='report_loss_per_steps', type=int, default=100,
                        help='(Optional) The number of steps per validation (default 100).')
    parser.add_argument('--lr', action="store", dest='lr', help='(Optional) Neural network learning rate.', type=float)
    parser.add_argument('--weight-decay', action="store", dest='weight_decay', help='(Optional) Neural network weight decay.', type=float, default=1e-4)
    parser.add_argument('--bs', action="store", dest='bs', help='(Optional) Neural network batch size.', type=int)
    parser.add_argument('--grad-accum-steps', action="store", dest='grad_accum_steps', type=int, default=16)
    parser.add_argument('--seed', action="store", dest='seed', type=int, default=1, help='(Optional) Random seed (default to 1).')
    parser.add_argument('--multimodal', action="store_true", dest='using_multimodal', help='using multimodal image features.')
    parser.add_argument('--save-checkpoint-per-epoch', action="store", dest='save_checkpoint_per_epoch', help='save checkpoint per epoch', type=int)
    parser.add_argument('--save-checkpoint-per-step', action="store", dest='save_checkpoint_per_step', help='save checkpoint per step', type=int)
    parser.add_argument('--rank', action="store", dest='rank', help='the rank of low rank matrices', type=int, default=-1)
    parser.add_argument('--resume-path', action="store", dest='resume_path', help='using for resume')
    parser.add_argument('--scheduled-sampling', action="store_true", dest='scheduled_sampling', help='using scheduled sampling, a common method to reduce exposure bias to improve '\
                                                                                                     'sequence generation by mixing teacher-forcing generation and auto-regressive generation. '\
                                                                                                     'see: https://www.activeloop.ai/resources/glossary/scheduled-sampling/')
    parser.add_argument('--mix-rate', action="store", dest='mix_rate', help='the mixing rate when using scheduled sampling', type=float, default=0.04)
    args = parser.parse_args()

    # handle defautl settings
    args.his_window = cfg.default_history_window if args.his_window is None else args.his_window
    args.fut_window = cfg.default_future_window if args.fut_window is None else args.fut_window
    args.trim_head = cfg.default_trim_head if args.trim_head is None else args.trim_head
    args.trim_tail = cfg.default_trim_tail if args.trim_tail is None else args.trim_tail
    args.dataset_frequency = cfg.default_dataset_frequency if args.dataset_frequency is None else args.dataset_frequency
    args.sample_step = cfg.default_sample_step if args.sample_step is None else args.sample_step
    args.epochs = cfg.default_epochs if args.epochs is None else args.epochs
    args.lr = cfg.default_lr if args.lr is None else args.lr
    args.weight_decay = cfg.default_weight_decay if args.weight_decay is None else args.weight_decay
    args.bs = cfg.default_bs if args.bs is None else args.bs
    args.grad_accum_steps = cfg.default_grad_accum_step if args.grad_accum_steps is None else args.grad_accum_steps
    args.steps_per_valid = cfg.default_steps_per_valid if args.steps_per_valid is None else args.steps_per_valid

    
    if args.device_out is None:  
        args.device_out = args.device

    if args.train_dataset is None:
        args.train_dataset = args.test_dataset
    if args.test_dataset is None:
        args.test_dataset = args.train_dataset

    print(args)
    run(args)
