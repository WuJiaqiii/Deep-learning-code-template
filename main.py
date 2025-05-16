import os
import argparse
import torch
import collections

from models import *
from data.data_loader import *

from utils.utils import create_logger, set_seed, Config

def get_parser():
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    ## dataset config
    parser.add_argument('--dataset_path', default='data/RML2016.10a_dict.pkl', type=str)

    ## train config
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', type=float, default=1e-6)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--save_interval', default=20, type=int)
    parser.add_argument('--early_stop_patience', default=200, type=int)

    ## other config
    parser.add_argument('--use_data_parallel', type=bool, default=True, help="Whether to use DataParallel for multi-GPU training")
    parser.add_argument('--use_amp_autocast', type=bool, default=False)
    
    args = parser.parse_args()
    
    return args

def main(args):
    
    set_seed(seed=42)
    config = Config(args)
    logger = create_logger(os.path.join(config.log_dir, f"train_log.log"))

    ## TODO:

if __name__ == "__main__":

    args = get_parser()
    main(args)