# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""All functions and modules related to model definition.
"""

import torch
import numpy as np
import logging

_MODELS = {}


def register_model(cls=None, *, name=None):
    """A decorator for registering model classes."""

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _MODELS:
            raise ValueError(f'Already registered model with name: {local_name}')
        _MODELS[local_name] = cls
        return cls
    #print(cls, name)
    if cls is None:
        return _register
    else:
        return _register(cls)


def get_model(name):
    #print(_MODELS)
    return _MODELS[name]


def get_sigmas(config):
    """Get sigmas --- the set of noise levels for SMLD from config files.
    Args:
        config: A ConfigDict object parsed from the config file
    Returns:
        sigmas: a jax numpy arrary of noise levels
    """
    sigmas = np.exp(
        np.linspace(np.log(config.model.sigma_max), np.log(config.model.sigma_min), config.model.num_scales))

    return sigmas


def get_ddpm_params(config):
    """Get betas and alphas --- parameters used in the original DDPM paper."""
    num_diffusion_timesteps = 1000
    # parameters need to be adapted if number of time steps differs from 1000
    beta_start = config.model.beta_min / config.model.num_scales
    beta_end = config.model.beta_max / config.model.num_scales
    betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)

    alphas = 1. - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)
    sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
    sqrt_1m_alphas_cumprod = np.sqrt(1. - alphas_cumprod)

    return {
        'betas': betas,
        'alphas': alphas,
        'alphas_cumprod': alphas_cumprod,
        'sqrt_alphas_cumprod': sqrt_alphas_cumprod,
        'sqrt_1m_alphas_cumprod': sqrt_1m_alphas_cumprod,
        'beta_min': beta_start * (num_diffusion_timesteps - 1),
        'beta_max': beta_end * (num_diffusion_timesteps - 1),
        'num_diffusion_timesteps': num_diffusion_timesteps
    }


def create_model(config):
    """Create the score model."""
    model_name = config.model.name
    score_model = get_model(model_name)(config)
    score_model = score_model.to(config.device)

    pytorch_total_grad_params = sum(p.numel() for p in score_model.parameters() if p.requires_grad)
    logging.info(f'total number of trainable parameters in the Score Model: {pytorch_total_grad_params}')
    pytorch_total_params = sum(p.numel() for p in score_model.parameters())
    logging.info(f'total number of parameters in the Score Model: {pytorch_total_params}')

    if config.world_size > 1:
        logging.info(f"using {config.world_size} GPUs!")
    score_model = torch.nn.DataParallel(score_model)
    return score_model

## UNet model creater
def create_model_edm(config):
    from .edm_networks import DhariwalUNet
    unet = DhariwalUNet(
                    img_resolution=config.data.image_size, 
                    in_channels=config.data.num_channels, 
                    out_channels=config.data.num_channels, 
                    label_dim=config.data.num_classes, # 1000 for ImageNet
                    model_channels=config.model.nf, 
                    channel_mult=config.model.ch_mult, 
                    num_blocks=config.model.num_res_blocks, 
                    attn_resolutions=config.model.attn_resolutions, 
                    dropout=0.13, 
                    )
    pytorch_total_grad_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    logging.info(f'total number of trainable parameters in the Score Model: {pytorch_total_grad_params}')
    pytorch_total_params = sum(p.numel() for p in unet.parameters())
    logging.info(f'total number of parameters in the Score Model: {pytorch_total_params}')
    return unet
    

def get_model_fn(model, train=False):
    """Create a function to give the output of the score-based model.

    Args:
        model: The score model.
        train: `True` for training and `False` for evaluation.

    Returns:
        A model function.
    """

    def model_fn(x, labels):
        """Compute the output of the score-based model.

        Args:
        x: A mini-batch of input data.
        labels: A mini-batch of conditioning variables for time steps. Should be interpreted differently
            for different models.

        Returns:
        A tuple of (model output, new mutable states)
        """
        if not train:
            model.eval()
            return model(x, labels)
        else:
            model.train()
            return model(x, labels)

    return model_fn


def to_flattened_numpy(x):
    """Flatten a torch tensor `x` and convert it to numpy."""
    return x.detach().cpu().numpy().reshape((-1,))


def from_flattened_numpy(x, shape):
    """Form a torch tensor with the given `shape` from a flattened numpy array `x`."""
    return torch.from_numpy(x.reshape(shape))

def load_mismatch_state_dict(model, old_state_dict):
    """load weight from checkpoint with mismatch size"""
    new_state_dict = model.state_dict()
    # Iterate over the model's parameters
    count = 0
    mis_count = 0
    not_count = 0
    for key in new_state_dict:
        if key in old_state_dict:
            # If the dimensions match, copy the weights
            if old_state_dict[key].size() == new_state_dict[key].size():
                new_state_dict[key] = old_state_dict[key]
            else:
                # Here you can handle the mismatch, e.g., log it, reinitialize, etc.
                print(f"Skipping {key} due to size mismatch.")
                mis_count += 1
        else:
            # Handle the missing keys, if necessary
            print(f"{key} is not found in the old model.")
            not_count += 1
        count += 1
    logging.info(f'skiped {mis_count} keys; {not_count} not found keys; {count} total keys')
    return new_state_dict