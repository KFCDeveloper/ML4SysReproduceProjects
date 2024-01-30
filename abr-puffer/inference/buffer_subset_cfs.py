import numpy as np
import torch
import datetime
import torch.nn as nn
import os
import argparse

from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument("--dir", help="root directory")
parser.add_argument("--C", type=float, help="discriminator loss coefficient")
parser.add_argument("--left_out_policy", type=str, help="left out policy")
parser.add_argument("--month", type=int, default=None)
parser.add_argument("--year", type=int, default=None)
parser.add_argument("--model_number", type=int, help="saved model epoch number", default=5000)
args = parser.parse_args()


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


class MLP(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_sizes, activation):
        super().__init__()
        self.predict = mlp(sizes=[input_dim] + list(hidden_sizes) + [output_dim], activation=activation,
                           output_activation=nn.Identity)

    def forward(self, raw_input):
        prediction = self.predict(raw_input)
        return prediction


class LinearBBA(object):
    def __init__(self, ssim_table, size_table, lower=3, upper=13.5):
        self.lower = lower
        self.upper = upper
        assert ssim_table.shape == size_table.shape
        self.num_formats = ssim_table.shape[1]
        self.ssim_table = ssim_table
        self.size_table = size_table

    def select_video_format(self, index_curr, buffer, no_ssim=False):
        invalid_mask = np.logical_or(np.isnan(self.size_table[index_curr]),
                                     np.isnan(self.ssim_table[index_curr]))
        size_arr_valid = np.ma.array(self.size_table[index_curr], mask=invalid_mask)
        ssim_arr_valid = np.ma.array(self.ssim_table[index_curr], mask=invalid_mask)
        min_choice = size_arr_valid.argmin()
        max_choice = size_arr_valid.argmax()
        if buffer < self.lower:
            act = min_choice
        elif buffer >= self.upper:
            act = max_choice
        else:
            ratio = (buffer - self.lower) / float(self.upper - self.lower)
            min_chunk = size_arr_valid[min_choice]
            max_chunk = size_arr_valid[max_choice]
            bitrate = ratio * (max_chunk - min_chunk) + min_chunk
            mask = np.logical_or(invalid_mask, size_arr_valid > bitrate)
            if no_ssim:
                act = np.ma.array(size_arr_valid, mask=mask).argmax()
            else:
                act = np.ma.array(ssim_arr_valid, mask=mask).argmax()
        return act, self.size_table[index_curr][act], self.ssim_table[index_curr][act]


class BolaBasic(object):
    size_ladder_bytes = [44319, 93355, 115601, 142904, 196884, 263965, 353752, 494902, 632193, 889893]
    ssim_index_ladder = [0.91050748, 0.94062527, 0.94806355, 0.95498943, 0.96214503, 0.96717277, 0.97273958, 0.97689813,
                         0.98004106, 0.98332605]
    MIN_BUF_S = 3
    MAX_BUF_S = 15
    MIN_SSIM = 0
    MAX_SSIM = 60
    chunk_length = 2.002

    def __init__(self, version, ssim_table, size_table):
        """

        :param version:
        :type version: int
        :type ssim_table: np.ndarray
        :type size_table: np.ndarray
        """
        assert self.size_ladder_bytes[0] < self.size_ladder_bytes[1]
        assert self.ssim_index_ladder[0] < self.ssim_index_ladder[1]
        assert self.MIN_BUF_S < self.MAX_BUF_S
        assert version in [1, 2]
        self.version = version
        assert ssim_table.shape == size_table.shape
        self.num_formats = ssim_table.shape[1]
        self.ssim_table = ssim_table
        self.size_table = size_table

        smallest = {'size': self.size_ladder_bytes[0],
                    'utility': self.utility(self.ssim_index_ladder[0])}
        second_smallest = {'size': self.size_ladder_bytes[1],
                           'utility': self.utility(self.ssim_index_ladder[1])}
        largest = {'size': self.size_ladder_bytes[-1],
                   'utility': self.utility(self.ssim_index_ladder[-1])}

        size_delta = self.size_ladder_bytes[1] - self.size_ladder_bytes[0]
        if version == 1:
            utility_high = largest['utility']
        else:
            utility_high = self.utility(1)

        size_utility_term = second_smallest['size'] * smallest['utility'] - smallest['size'] * \
                            second_smallest['utility']
        gp_nominator = self.MAX_BUF_S * size_utility_term - utility_high * self.MIN_BUF_S * size_delta
        gp_denominator = ((self.MIN_BUF_S - self.MAX_BUF_S) * size_delta)
        self.gp = gp_nominator / gp_denominator
        self.Vp = self.MAX_BUF_S / (utility_high + self.gp)

    def utility(self, ssim_index):
        """

        :param ssim_index:
        :type ssim_index: float or np.ndarray
        :return:
        :rtype: float or np.ndarray
        """

        if self.version == 1:
            return np.where(ssim_index == 1, self.MAX_SSIM, np.clip(-10 * np.log10(1 - ssim_index),
                                                                    a_min=self.MIN_SSIM,
                                                                    a_max=self.MAX_SSIM))
        else:
            return ssim_index

    def objective(self, utility, size, buffer_in_chunks):
        """

        :param utility:
        :type utility: float or np.ndarray
        :param size:
        :type size: float or np.ndarray
        :param buffer_in_chunks:
        :type buffer_in_chunks: float
        :return:
        :rtype: float or np.ndarray
        """
        return (self.Vp / self.chunk_length * (utility + self.gp) - buffer_in_chunks) / size

    def choose_max_objective(self, format_sizes, format_ssims, buffer_in_chunks):
        """

        :param format_sizes:
        :type format_sizes: np.ndarray
        :param format_ssims:
        :type format_ssims: np.ndarray
        :param buffer_in_chunks:
        :type buffer_in_chunks: float
        :return:
        :rtype: (int, float, float, float)
        """
        objs = self.objective(self.utility(format_ssims), format_sizes, buffer_in_chunks)
        chosen_index = np.argmax(objs)
        return chosen_index, format_sizes[chosen_index], format_ssims[chosen_index], objs[chosen_index]

    def choose_max_scaled_utility(self, format_sizes, format_ssims):
        """

        :param format_sizes:
        :type format_sizes: np.ndarray
        :param format_ssims:
        :type format_ssims: np.ndarray
        :return:
        :rtype: (int, float, float)
        """
        chosen_index = np.argmax(self.utility(format_ssims) + self.gp)
        return chosen_index, format_sizes[chosen_index], format_ssims[chosen_index]

    def select_video_format(self, index_curr, buffer):
        """

        :param index_curr:
        :type index_curr: int
        :param buffer:
        :type buffer: float
        :return: Return the action index, size and ssim
        :rtype: (int, float, float)
        """
        buffer_in_chunks = buffer / self.chunk_length
        valid_mask = np.logical_not(np.logical_or(np.isnan(self.size_table[index_curr]),
                                                  np.isnan(self.ssim_table[index_curr])))
        size_arr_valid = self.size_table[index_curr][valid_mask]
        ssim_arr_valid = self.ssim_table[index_curr][valid_mask]
        index_arr_valid = np.arange(self.num_formats)[valid_mask]

        max_obj_index, max_obj_size, max_obj_ssim, max_obj = self.choose_max_objective(size_arr_valid,
                                                                                       ssim_arr_valid,
                                                                                       buffer_in_chunks)

        if self.version == 1 or max_obj >= 0:
            return index_arr_valid[max_obj_index], max_obj_size, max_obj_ssim
        else:
            max_util_index, max_util_size, max_util_ssim = self.choose_max_scaled_utility(size_arr_valid,
                                                                                          ssim_arr_valid)
            return index_arr_valid[max_util_index], max_util_size, max_util_ssim


def counterfactual(latent_list, abr_algo, predictor, buf_mean, buf_std, next_buf_mean, next_buf_std,
                   size_mean, size_std):
    buffer = 0
    buf_history = [buffer]
    size_history = []
    ssim_history = []
    for step in range(len(latent_list)):
        _, selected_size, selected_ssim = abr_algo.select_video_format(step, buffer)
        size_history.append(selected_size)
        ssim_history.append(selected_ssim)
        selected_size_white = (selected_size - size_mean) / size_std
        buffer_white = (buffer - buf_mean) / buf_std
        input_numpy = np.array([buffer_white, selected_size_white])
        input_numpy = np.expand_dims(input_numpy, axis=0)
        input_tensor = torch.as_tensor(input_numpy, dtype=torch.float32, device=torch.device('cpu'))
        feature = latent_list[step]
        feature_tensor = torch.as_tensor(feature, dtype=torch.float32, device=torch.device('cpu'))
        input_tensor = torch.cat((input_tensor, feature_tensor), dim=1)
        with torch.no_grad():
            next_buffer_white_tensor = predictor(input_tensor)
        next_buffer_white = next_buffer_white_tensor[0, 0].cpu().numpy()
        next_buffer = (next_buffer_white * next_buf_std) + next_buf_mean
        next_buffer = max(0, next_buffer)
        if step < len(latent_list) - 1:
            buf_history.append(next_buffer)
        buffer = next_buffer
    assert len(buf_history) == len(latent_list) == len(size_history) == len(ssim_history)
    return buf_history, size_history, ssim_history


DISCRIMINATOR_EPOCH = 10
left_out_text = f'_{args.left_out_policy}'
PERIOD_TEXT = f'2020-07-27to2021-06-01{left_out_text}'
C = args.C
cf_path = f'{args.dir}{PERIOD_TEXT}_buff_cfs/inner_loop_{DISCRIMINATOR_EPOCH}/C_{C}/cfs/model_{args.model_number}'
os.makedirs(cf_path, exist_ok=True)

data_path = f'{args.dir}subset_data/{args.left_out_policy}'
buffer_mean = np.load(f'{data_path}/buffs_mean.npy')
next_buffer_mean = np.load(f'{data_path}/next_buffs_mean.npy')
buffer_std = np.load(f'{data_path}/buffs_std.npy')
next_buffer_std = np.load(f'{data_path}/next_buffs_std.npy')
action_mean = np.load(f'{data_path}/actions_mean.npy')
action_std = np.load(f'{data_path}/actions_std.npy')

start_date = datetime.date(2020, 7, 27)
end_date = datetime.date(2021, 6, 1)
all_days = [start_date + datetime.timedelta(days=x) for x in range((end_date - start_date).days + 1)]
all_days = [day for day in all_days if day not in [datetime.date(2019, 5, 12), datetime.date(2019, 5, 13),
                                                   datetime.date(2019, 5, 15), datetime.date(2019, 5, 17),
                                                   datetime.date(2019, 5, 18), datetime.date(2019, 5, 19),
                                                   datetime.date(2019, 5, 25), datetime.date(2019, 5, 27),
                                                   datetime.date(2019, 5, 30), datetime.date(2019, 6, 1),
                                                   datetime.date(2019, 6, 2), datetime.date(2019, 6, 3),
                                                   datetime.date(2019, 7, 2), datetime.date(2019, 7, 3),
                                                   datetime.date(2019, 7, 4), datetime.date(2020, 7, 7),
                                                   datetime.date(2020, 7, 8), datetime.date(2021, 6, 2),
                                                   datetime.date(2021, 6, 2), datetime.date(2021, 6, 3),
                                                   datetime.date(2022, 1, 31), datetime.date(2022, 2, 1),
                                                   datetime.date(2022, 2, 2), datetime.date(2022, 2, 3),
                                                   datetime.date(2022, 2, 4), datetime.date(2022, 2, 5),
                                                   datetime.date(2022, 2, 6), datetime.date(2022, 2, 7)]]

if args.month is not None and args.year is not None:
    all_days = [date for date in all_days if date.month == args.month and date.year == args.year]

model_path = f'{args.dir}{PERIOD_TEXT}_trained_models/inner_loop_{DISCRIMINATOR_EPOCH}/C_{C}'
predictor = torch.load(f"{model_path}/{args.model_number}_predictor.pth", map_location=torch.device('cpu')).cpu()
cooked_path = f'{args.dir}cooked'
latent_path = f'{args.dir}{PERIOD_TEXT}_features/inner_loop_{DISCRIMINATOR_EPOCH}/C_{C}/model_{args.model_number}'

for today in tqdm(all_days):
    date_string = "%d-%02d-%02d" % (today.year, today.month, today.day)
    trajs = np.load(f"{cooked_path}/{date_string}_trajs.npy", allow_pickle=True)
    latent_list = np.load(f'{latent_path}/{date_string}_features.npy', allow_pickle=True)
    bba_buffs = []
    bola1_buffs = []
    bola2_buffs = []
    bba_sizes = []
    bola1_sizes = []
    bola2_sizes = []
    bba_ssims = []
    bola1_ssims = []
    bola2_ssims = []
    for idx, traj in enumerate(trajs):
        latents = latent_list[idx]
        linear_bba = LinearBBA(ssim_table=traj[:, 28:40], size_table=traj[:, 16:28])
        bola1 = BolaBasic(1, ssim_table=traj[:, 28:40], size_table=traj[:, 16:28])
        bola2 = BolaBasic(2, ssim_table=traj[:, 28:40], size_table=traj[:, 16:28])
        bba_buf_history, bba_size_history, bba_ssim_history = counterfactual(
            latents, linear_bba, predictor, buffer_mean, buffer_std,
            next_buffer_mean, next_buffer_std, action_mean, action_std)
        bba_buffs.append(bba_buf_history)
        bba_sizes.append(bba_size_history)
        bba_ssims.append(bba_ssim_history)
        bola1_buf_history, bola1_size_history, bola1_ssim_history = counterfactual(
            latents, bola1, predictor, buffer_mean, buffer_std,
            next_buffer_mean, next_buffer_std, action_mean, action_std)
        bola1_buffs.append(bola1_buf_history)
        bola1_sizes.append(bola1_size_history)
        bola1_ssims.append(bola1_ssim_history)
        bola2_buf_history, bola2_size_history, bola2_ssim_history = counterfactual(
            latents, bola2, predictor, buffer_mean, buffer_std,
            next_buffer_mean, next_buffer_std, action_mean, action_std)
        bola2_buffs.append(bola2_buf_history)
        bola2_sizes.append(bola2_size_history)
        bola2_ssims.append(bola2_ssim_history)
    np.save(f'{cf_path}/{date_string}_linear_bba_buffs.npy', np.array(bba_buffs, dtype=object))
    np.save(f'{cf_path}/{date_string}_bola1_buffs.npy', np.array(bola1_buffs, dtype=object))
    np.save(f'{cf_path}/{date_string}_bola2_buffs.npy', np.array(bola2_buffs, dtype=object))
    np.save(f'{cf_path}/{date_string}_linear_bba_actions.npy', np.array(bba_sizes, dtype=object))
    np.save(f'{cf_path}/{date_string}_bola1_actions.npy', np.array(bola1_sizes, dtype=object))
    np.save(f'{cf_path}/{date_string}_bola2_actions.npy', np.array(bola2_sizes, dtype=object))
    np.save(f'{cf_path}/{date_string}_linear_bba_ssims.npy', np.array(bba_ssims, dtype=object))
    np.save(f'{cf_path}/{date_string}_bola1_ssims.npy', np.array(bola1_ssims, dtype=object))
    np.save(f'{cf_path}/{date_string}_bola2_ssims.npy', np.array(bola2_ssims, dtype=object))