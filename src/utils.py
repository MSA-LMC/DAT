import numpy as np
import random
import torch
import os
import shutil

def set_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def copy_and_save_file(save_dir):
    current_path = os.path.abspath(__file__)
    current_file_name = os.path.basename(__file__)
    backup_folder = f'./output_model/{save_dir}'
    backup_file_path = os.path.join(backup_folder, current_file_name)
    shutil.copyfile(current_path, backup_file_path)
