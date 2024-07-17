import os
join = os.path.join
from typing import Iterator
import torch
from torch import Tensor, nn
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import save_image, make_grid
import subprocess
import shutil
import torch
import os
import logging
import random


def seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True


def restore_checkpoint(ckpt_dir, state, device):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir, exist_ok=True)
        logging.warning(f"No checkpoint found at {ckpt_dir}. "
                        f"Returned the same state as input")
        return state
    else:
        loaded_state = torch.load(ckpt_dir, map_location=device)
        state['optimizer'].load_state_dict(loaded_state['optimizer'])
        state['model'].load_state_dict(loaded_state['model'], strict=False)
        state['ema'].load_state_dict(loaded_state['ema'])
        state['step'] = loaded_state['step']
        return state


def save_checkpoint(ckpt_dir, state):
    saved_state = {
        'optimizer': state['optimizer'].state_dict(),
        'model': state['model'].state_dict(),
        'ema': state['ema'].state_dict(),
        'step': state['step']
    }
    torch.save(saved_state, ckpt_dir)


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def save_image_batch(batch, img_size, sample_path, log_name="examples"):
        sample_grid = make_grid(batch, nrow=int(np.ceil(np.sqrt(batch.shape[0]))), padding=img_size // 16)
        save_image(sample_grid, join(sample_path, log_name))


def update_curve(values, label, x_label, work_path, run_id):
    fig, ax = plt.subplots()
    ax.plot(values, label=label)
    ax.set_xlabel(x_label)
    ax.set_ylabel(label)
    ax.legend()
    plt.savefig(f'{work_path}/{label}_curve_{run_id}.png', dpi=600)
    plt.close()


def get_file_list():
    return [
        b.decode()
        for b in set(
            subprocess.check_output(
                'git ls-files -- ":!:load/*"', shell=True
            ).splitlines()
        )
        | set(  # hard code, TODO: use config to exclude folders or files
            subprocess.check_output(
                "git ls-files --others --exclude-standard", shell=True
            ).splitlines()
        )
    ]


def save_code_snapshot(model_path):
    os.makedirs(model_path, exist_ok=True)
    for f in get_file_list():
        if not os.path.exists(f) or os.path.isdir(f):
            continue
        os.makedirs(os.path.join(model_path, os.path.dirname(f)), exist_ok=True)
        shutil.copyfile(f, os.path.join(model_path, f))
