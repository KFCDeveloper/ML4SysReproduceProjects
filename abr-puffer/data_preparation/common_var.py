# coding=utf-8
import datetime
import torch

class Durration_CON:
    start_date = datetime.date(2020, 3, 27) # 2020, 7, 27; 2020, 9, 27; 2020, 11, 27; 2021, 1, 27; 2021, 3, 27;
    end_date = datetime.date(2021, 6, 1)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# use the input dir to control Durration_CON 
def match_date(args):
    dir_date = {"CAUSALSIM_DIR-21-3-27_copy/":[datetime.date(2021, 3, 27),datetime.date(2021, 6, 1)],
                "CAUSALSIM_DIR-20-9-27-6monthmodel/":[datetime.date(2020, 9, 27),datetime.date(2021, 6, 1)],
                "CAUSALSIM_DIR-20-7-27/":[datetime.date(2020, 7, 27),datetime.date(2021, 6, 1)],
                "CAUSALSIM_DIR-20-9-27/":[datetime.date(2020, 9, 27),datetime.date(2021, 6, 1)],
                "CAUSALSIM_DIR-20-11-27/":[datetime.date(2020, 11, 27),datetime.date(2021, 6, 1)],
                "CAUSALSIM_DIR-21-1-27/":[datetime.date(2021, 1, 27),datetime.date(2021, 6, 1)],
                "CAUSALSIM_DIR-21-3-27/":[datetime.date(2021, 3, 27),datetime.date(2021, 6, 1)],
                #
                "CAUSALSIM_DIR-20-7-27to20-9-27/":[datetime.date(2020, 7, 27),datetime.date(2020, 9, 27)],
                "CAUSALSIM_DIR-20-9-27to20-11-27/":[datetime.date(2020, 9, 27),datetime.date(2020, 11, 27)],
                "CAUSALSIM_DIR-20-11-27_copy/":[datetime.date(2020, 11, 27),datetime.date(2021, 6, 1)],
                "CAUSALSIM_DIR-20-7-27to20-9-27_total_view/":[datetime.date(2020, 7, 27),datetime.date(2020, 9, 27)],
                "CAUSALSIM_DIR-20-9-27to20-11-27_total_view/":[datetime.date(2020, 9, 27),datetime.date(2020, 11, 27)],
                # each month 7,8,9,10,11,12,1,2
                # {"CAUSALSIM_DIR-dis-20-7-27/","CAUSALSIM_DIR-dis-20-8-27/","CAUSALSIM_DIR-dis-20-9-27/","CAUSALSIM_DIR-dis-20-10-27/","CAUSALSIM_DIR-dis-20-11-27/","CAUSALSIM_DIR-dis-20-12-27/","CAUSALSIM_DIR-dis-21-1-27/","CAUSALSIM_DIR-dis-21-2-27/"}
                "CAUSALSIM_DIR-dis-20-7-27/":[datetime.date(2020, 7, 27),datetime.date(2020, 8, 27)],
                "CAUSALSIM_DIR-dis-20-8-27/":[datetime.date(2020, 8, 27),datetime.date(2020, 9, 27)],
                "CAUSALSIM_DIR-dis-20-9-27/":[datetime.date(2020, 9, 27),datetime.date(2020, 10, 27)],
                "CAUSALSIM_DIR-dis-20-10-27/":[datetime.date(2020, 10, 27),datetime.date(2020, 11, 27)],
                "CAUSALSIM_DIR-dis-20-11-27/":[datetime.date(2020, 11, 27),datetime.date(2020,12, 27)],
                "CAUSALSIM_DIR-dis-20-12-27/":[datetime.date(2020, 12, 27),datetime.date(2021, 1, 27)],
                "CAUSALSIM_DIR-dis-21-1-27/":[datetime.date(2021, 1, 27),datetime.date(2021, 2, 27)],
                "CAUSALSIM_DIR-dis-21-2-27/":[datetime.date(2021, 2, 27),datetime.date(2021, 3, 27)],
                }
    if args.dir in dir_date:
        Durration_CON.start_date = dir_date[args.dir][0]
        Durration_CON.end_date = dir_date[args.dir][1]