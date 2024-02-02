# coding=utf-8
import datetime
import torch

class Durration_CON:
    start_date = datetime.date(2021, 3, 1) # 2020, 7, 27; 2020, 9, 27; 2020, 11, 27; 2021, 1, 27; 2021, 3, 27;
    end_date = datetime.date(2021, 6, 1)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")