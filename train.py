import os
# import sys
join = os.path.join
from absl import app
from absl import flags
from ml_collections.config_flags import config_flags
import logging
import torch
import numpy as np
from models import utils as mutils
from models.ema import ExponentialMovingAverage
# Keep the import below for registering all model definitions
from models import ddpm, ncsnv2, ncsnpp
import reflow.datasets as datasets
from reflow.utils import (  seed_everywhere,
                            save_code_snapshot,
                            save_checkpoint,
                            restore_checkpoint,
                            save_image_batch)
from reflow import RectifiedFlow
from reflow import losses as losses
from reflow import sampling as sampling
from reflow.augment import AugmentPipe
from datetime import datetime


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
  "config", None, "Training configuration.", lock_config=True)
flags.mark_flags_as_required(["config"])

def main(argv):
    config = FLAGS.config
    
    ### set random seed everywhere
    seed_everywhere(config.seed)

    ### set up paths
    run_id = datetime.now().strftime("%Y%m%d_%H%M")
    work_path = join(config.work_dir, f'{config.expr}/{run_id}')
    checkpoint_dir = join(work_path, 'checkpoints')
    checkpoint_meta_dir = os.path.join(work_path, "checkpoints-meta", "checkpoint.pth")
    sample_dir = join(work_path, 'samples')
    os.makedirs(work_path, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.dirname(checkpoint_meta_dir), exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)
    save_code_snapshot(join(work_path, f'codes'))

    ### set up logger
    gfile_stream = open(f'{work_path}/std_{run_id}.log', 'w')
    handler = logging.StreamHandler(gfile_stream)
    formatter = logging.Formatter('%(filename)s - %(asctime)s - %(levelname)s --> %(message)s')
    handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler)
    logger.setLevel('INFO')
    
    ### basic info
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config.device = device
    logger.info(f'Using device: {device}; version: {str(torch.version.cuda)}')
    if device.type == 'cuda':
        logger.info(torch.cuda.get_device_name(0))

    ### data loader
    data_loader = datasets.get_dataset(config)
    logger.info(f'length of dataloader: {len(data_loader)}')

    ### data augmentation
    if config.data.use_aug:
        # augment_pipe only used in the training part
        # augment_pipe = AugmentPipe(p=0.12, xflip=1e8, yflip=1, scale=1, rotate_frac=1, aniso=1, translate_frac=1)
        augment_pipe = AugmentPipe(p=0.12, xflip=1e8, yflip=1)  # 50% xflip
    else:
        augment_pipe = None

    ### create model & optimizer
    score_model = mutils.create_model(config) if config.model.name != 'DhariwalUNet' else mutils.create_model_edm(config)
    logger.info("#################### Model: ####################")
    logger.info(f'initialize model')
    # logger.info(f'{score_model}')
    score_model.to(device)
    ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
    optimizer = losses.get_optimizer(config, score_model.parameters())
    state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

    ### Resume training when intermediate checkpoints are detected
    if config.training.resume_from:
        assert config.flow.pre_train_model == '', "no need pre_train_model for resume_from training"
        checkpoint_meta_dir = os.path.join(config.training.resume_from, "checkpoints-meta", "checkpoint.pth")
        assert os.path.exists(checkpoint_meta_dir), f"Checkpoint meta file {checkpoint_meta_dir} does not exist"
        state = restore_checkpoint(checkpoint_meta_dir, state, config.device)
        logger.info(f"Resuming training from checkpoint {checkpoint_meta_dir}")
        checkpoint_meta_dir = os.path.join(work_path, "checkpoints-meta", "checkpoint.pth")
    initial_step = int(state['step'])

    ### Load pre-trained model if specified: for finetuning
    if config.flow.pre_train_model:
        # only load the score_model parameters
        checkpoint = torch.load(config.flow.pre_train_model, map_location=device)
        try:
            score_model.load_state_dict(checkpoint['model'], strict=False)
        except:
            checkpoint['model'] = mutils.load_mismatch_state_dict(score_model, checkpoint['model'])
            score_model.load_state_dict(checkpoint['model'], strict=False)
        ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
        # ema.load_state_dict(checkpoint['ema'])
        # ema.copy_to(score_model.parameters())
        # ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
        config.optim.warmup = 0     # no warmup for finetuning
        optimizer = losses.get_optimizer(config, score_model.parameters())
        state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)
        flow = RectifiedFlow(model=score_model, ema_model=ema, cfg=config)
        logger.info(f'loaded model from path: {config.flow.pre_train_model}')
        del checkpoint
        torch.cuda.empty_cache()
    else:
        flow = RectifiedFlow(model=score_model, ema_model=ema, cfg=config)
    flow.model.train()

    ### create teacher model for distillation
    if config.flow.use_teacher:
        score_model_teacher = mutils.create_model(config) if config.model.name != 'DhariwalUNet' else mutils.create_model_edm(config)
        checkpoint_teacher = torch.load(config.flow.pre_train_model, map_location=device)
        score_model_teacher.load_state_dict(checkpoint_teacher['model'], strict=False)
        flow.model_teacher = score_model_teacher.to(device)
        ema_teacher = ExponentialMovingAverage(score_model_teacher.parameters(), decay=config.model.ema_rate)
        ema_teacher.load_state_dict(checkpoint_teacher['ema'])
        ema_teacher.copy_to(flow.model_teacher.parameters())
        logging.info(f'init teacher model')
        del checkpoint_teacher
        flow.model_teacher.eval()

    ### Create data normalizer and its inverse
    scaler = datasets.get_data_scaler(config)
    inverse_scaler = datasets.get_data_inverse_scaler(config)

    ### Building sampling functions
    if config.training.snapshot_sampling:
        sampling_shape = (config.sampling.batch_size, config.data.num_channels,
                        config.data.image_size, config.data.image_size)
        sampling_fn = sampling.get_flow_sampler(flow, sampling_shape, inverse_scaler, device=device)
        if config.sampling.sample_N != 1:
            sampling_fn_n1 = sampling.get_flow_sampler(flow, sampling_shape, inverse_scaler, device=device, use_ode_sampler='one_step')

    #################### train ####################
    optimize_fn = losses.optimization_manager(config)
    train_loss_values = []
    if config.training.progress:
        from tqdm import tqdm
        pbar = tqdm(range(config.training.n_iters))
    else:
        pbar = range(config.training.n_iters)
    for global_step in pbar:
        if global_step < initial_step:
            continue
        optimizer.zero_grad()
        batch_loss = torch.tensor(0.0, device=device)
        ### accumulation steps
        for _ in range(config.training.accumulation_steps):
            try:
                batch, label_dic = next(data_iterator)
            except:
                data_iterator = iter(data_loader)
                batch, label_dic = next(data_iterator)
            label = label_dic.to(device) if config.data.num_classes > 0 else None
            # perform a train step
            loss = flow.train_step(batch.to(device), global_step, augment_pipe, label=label)
            loss /= config.training.accumulation_steps
            loss.backward()
            batch_loss += loss
        
        ### post train step
        optimize_fn(optimizer, flow.model.parameters(), step=state['step'])
        state['step'] += 1
        # state['ema'].decay = flow.ema_decay_rate_schedule()   # for consistency model
        state['ema'].update(flow.model.parameters())

        train_loss_values.append(batch_loss.item())
        pbar.set_description(f"loss: {batch_loss.item()}") if config.training.progress else None

        ### record metric
        if global_step % config.training.record_iters == 0 and global_step != 0:
            # record
            current_lr = optimizer.param_groups[0]['lr']
            logger.info(f'step: {global_step:08d}; current lr: {current_lr:0.6f}; average loss: {np.average(train_loss_values):0.10f}; batch loss: {batch_loss.item():0.10f}')
            # save the training loss curve
            np.save(os.path.join(work_path, f"loss_values"), train_loss_values)
        
        ### Save a temporary checkpoint to resume training after pre-emption periodically
        if global_step % config.training.snapshot_freq_for_preemption == 0 and global_step != 0:
            save_checkpoint(checkpoint_meta_dir, state)

        ### save model
        if config.training.snapshot_freq and global_step % config.training.snapshot_freq == 0 and global_step != 0:
            # Save the checkpoint.
            save_step = global_step // config.training.snapshot_freq
            save_checkpoint(os.path.join(checkpoint_dir, f'checkpoint_{save_step}.pth'), state)
        
        ### Generate and save samples
        if config.training.snapshot_sampling > 0 and global_step % config.training.snapshot_sampling == 0 and global_step != 0:
            ema.store(score_model.parameters())
            score_model.eval()
            ema.copy_to(score_model.parameters())
            z0 = flow.get_z0(torch.zeros(sampling_shape, device=device), train=False).to(device)
            class_labels = None
            if config.data.num_classes:
                class_labels = torch.eye(config.data.num_classes, device=device)[torch.randint(0, config.data.num_classes, (config.sampling.batch_size,))]
            class_idx = None
            if class_idx is not None:
                class_labels[:, :] = 0
                class_labels[:, class_idx] = 1
            sample, n = sampling_fn(score_model, z=z0, label=class_labels)
            if config.sampling.sample_N != 1:
                sample_n1, _ = sampling_fn_n1(score_model, z=z0, label=class_labels)
            ema.restore(score_model.parameters())
            score_model.train()
            this_sample_dir = os.path.join(sample_dir, "iter_{}".format(global_step))
            os.makedirs(this_sample_dir, exist_ok=True)
            save_image_batch(sample, config.data.image_size, this_sample_dir, log_name=f'sample_steps_{n}.png')
            save_image_batch(sample_n1, config.data.image_size, this_sample_dir, log_name=f'sample_steps_1.png')
            np.save(os.path.join(this_sample_dir, f"sample_steps_{n}"), sample.cpu())
            np.save(os.path.join(this_sample_dir, f"sample_steps_1"), sample_n1.cpu())

        ### GPU usage at initial step
        if global_step == initial_step:
            global_free, total_gpu = torch.cuda.mem_get_info(0)
            logger.info(f'global free and total GPU memory: {round(global_free/1024**3,6)} GB, {round(total_gpu/1024**3,6)} GB')
            del global_free; del total_gpu
            logger.info("#################### Training Logs: ####################")


if __name__ == "__main__":
    app.run(main)
