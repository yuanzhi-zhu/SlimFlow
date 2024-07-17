import os
import sys
join = os.path.join
from absl import app
from absl import flags
from ml_collections.config_flags import config_flags
import torch
import numpy as np
from tqdm import tqdm
import logging
from models import utils as mutils
from models.ema import ExponentialMovingAverage
# Keep the import below for registering all model definitions
from models import ddpm, ncsnv2, ncsnpp
import reflow.datasets as datasets
from reflow.utils import restore_checkpoint, seed_everywhere
from reflow import RectifiedFlow, AnalyticFlow
from reflow import losses as losses
from reflow import sampling as sampling

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
  "config", None, "Training configuration.", lock_config=True)
flags.DEFINE_string("ckpt_path", None, "checkpoint path.")
flags.DEFINE_string("data_root", None, "data root path.")
flags.DEFINE_string("z0_path", None, "z0 data path.")
flags.DEFINE_string("z1_path", None, "z1 data path.")
flags.mark_flags_as_required(["config", "ckpt_path", "data_root"])

def save_data_pair(data_root, z0_cllt, z1_cllt, total_number_of_samples, z0_name='z0.npy', z1_name='z1.npy', class_cllt=None):
    z0_cllt = torch.cat(z0_cllt).cpu()[:total_number_of_samples]
    z1_cllt = torch.cat(z1_cllt).cpu()[:total_number_of_samples]
    logging.info(f'z1 shape: {z1_cllt.shape}; z0 shape: {z0_cllt.shape}')
    logging.info(f'z0 mean: {z0_cllt.mean()}, z0 std: {z0_cllt.std()}')
    if not os.path.exists(data_root):
        os.mkdir(data_root)
    np.save(os.path.join(data_root, z1_name), z1_cllt.numpy())
    np.save(os.path.join(data_root, z0_name), z0_cllt.numpy())
    if class_cllt is not None and len(class_cllt) > 0:
        class_cllt = torch.cat(class_cllt).cpu()[:total_number_of_samples].float()
        np.save(os.path.join(data_root, 'label.npy'), class_cllt.numpy())

def delete_tmp_data(data_root):
    # remove tmp data if exists
    if os.path.exists(os.path.join(data_root, 'z0_tmp.npy')):
        os.remove(os.path.join(data_root, 'z0_tmp.npy'))
    if os.path.exists(os.path.join(data_root, 'z1_tmp.npy')):
        os.remove(os.path.join(data_root, 'z1_tmp.npy'))

def load_tmp_data(data_root):
    z0_tmp_loaded = np.load(os.path.join(data_root, 'z0_tmp.npy'))
    z0_tmp_loaded = torch.from_numpy(z0_tmp_loaded)
    z1_tmp_loaded = np.load(os.path.join(data_root, 'z1_tmp.npy'))
    z1_tmp_loaded = torch.from_numpy(z1_tmp_loaded)
    return z0_tmp_loaded, z1_tmp_loaded

def main(argv):
    config = FLAGS.config
    ### set random seed everywhere
    seed_everywhere(config.seed)
    
    data_root = FLAGS.data_root 
    seeded_data_root = os.path.join(data_root, str(config.seed))
    os.makedirs(seeded_data_root, exist_ok=True)
    # set up logger
    gfile_stream = open(f'{seeded_data_root}/stdout.log', 'a+')
    handler = logging.StreamHandler(gfile_stream)
    formatter = logging.Formatter('%(filename)s - %(asctime)s - %(levelname)s --> %(message)s')
    handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler)
    logger.setLevel('INFO')
    logging.info(f'DATA PATH: {seeded_data_root}')
    
    ### basic info
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config.device = device
    logging.info(f'Using device: {device}; version: {str(torch.version.cuda)}')
    if device.type == 'cuda':
        logging.info(f'{torch.cuda.get_device_name(0)}')
    
    ### create model & optimizer
    # Initialize model.
    score_model = mutils.create_model(config) if config.model.name != 'DhariwalUNet' else mutils.create_model_edm(config)
    score_model.to(device)

    ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
    optimizer = losses.get_optimizer(config, score_model.parameters())
    state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

    ckpt_path = FLAGS.ckpt_path
    # Load checkpoint
    if config.sampling.direction == 'random_paired':
        logging.info('random paired data, no need to load model')
    elif 'gt_v' in config.sampling.direction:
        logging.info('gt velocity data, no need to load model')
    else:
        state = restore_checkpoint(ckpt_path, state, device=config.device)
        logging.info(f"load model from {ckpt_path}")
        ema.copy_to(score_model.parameters())
    flow = RectifiedFlow(model=score_model, ema_model=ema, cfg=config)
    flow.model.eval()

    # Create data normalizer and its inverse
    scaler = datasets.get_data_scaler(config)
    inverse_scaler = datasets.get_data_inverse_scaler(config)

    # Building sampling functions
    sampling_shape = (config.training.batch_size, config.data.num_channels,
                    config.data.image_size, config.data.image_size)
    sampling_fn = sampling.get_flow_sampler(flow, sampling_shape, inverse_scaler, device=device)

    ### reset random seed everywhere
    seed_everywhere(config.seed)  
    
    if 'gt_v' in config.sampling.direction:
        flow.T = flow.T - flow.eps  # necessary for (fast) convergence of rk45 sampler
        # gt points & particle initialization
        ###################### load datasets ######################
        dataloader = datasets.get_dataset(config)
        logger.info(f'length of dataloader: {len(dataloader)}')
        # loop dataloader
        all_images = []
        logging.info('loading dataset for gt velocity model')
        # Iterate through the dataloader to access batches of images as tensors
        for batch_images, batch_labels in tqdm(dataloader):
            if config.data.reflow_data_root:
                batch_images, z0 = torch.split(batch_images, 1, dim=1)
                batch_images, z0 = batch_images.squeeze(1), z0.squeeze(1)
            # Perform any additional processing if needed
            # For example, you can directly use batch_images in your model for inference or training
            all_images.append(batch_images)
        batch_tensor = torch.cat(all_images, dim=0)
        gt_images = batch_tensor.to(device)
        if hasattr(config.sampling, 'train_subset') and 'random' in config.sampling.train_subset:
            sample_number = int(config.sampling.train_subset.split('_')[-1])
            logging.info(f'randomly sample {sample_number} images from training set')
            gt_images = gt_images[torch.randperm(gt_images.shape[0])[:sample_number]]
        
        # analytic v model
        logging.info('create analytic velocity model')
        analytic_v = AnalyticFlow(gt_images).to(device)
    
    if 'from_z0' in config.sampling.direction:
        logging.info(f'Start generating data with ODE from z0, SEED: {config.seed}')
        
        if FLAGS.z0_path:
            z0_loaded = np.load(FLAGS.z0_path)
            z0_loaded = torch.from_numpy(z0_loaded).to(config.device)
            logging.info(f'loaded z0 shape: {z0_loaded.shape} from {FLAGS.z0_path}')

        ema.copy_to(score_model.parameters())
        data_cllt = []
        z0_cllt = []
        label_cllt = []
        nfes = []
        # if z1_tmp and z0_tmp exist, load them
        if os.path.exists(os.path.join(seeded_data_root, 'z1_tmp.npy')) and os.path.exists(os.path.join(seeded_data_root, 'z0_tmp.npy')):
            z0_tmp_loaded, data_tmp_loaded = load_tmp_data(seeded_data_root)
            data_cllt.append(data_tmp_loaded)
            z0_cllt.append(z0_tmp_loaded)
            assert data_tmp_loaded.shape[0] == z0_tmp_loaded.shape[0]
            current_loaded_number = data_tmp_loaded.shape[0] // config.training.batch_size
            logging.info(f'current z0_tmp loaded shape: {z0_tmp_loaded.shape}, current data_tmp loaded shape: {data_tmp_loaded.shape}')
        else:
            current_loaded_number = 0
        total_number_of_samples = config.sampling.total_number_of_samples
        num_iter = int(np.ceil(total_number_of_samples / config.training.batch_size))
        pbar = tqdm(range(num_iter))
        for data_step in pbar:
            if FLAGS.z0_path:
                z0 = z0_loaded[data_step*config.training.batch_size:(data_step+1)*config.training.batch_size]
            else:
                z0 = flow.get_z0(torch.zeros(sampling_shape, device=config.device), train=False).to(config.device)
            class_labels = None
            if config.data.num_classes:
                class_labels = torch.eye(config.data.num_classes, device=device)[torch.randint(0, config.data.num_classes, (config.training.batch_size,))]
                class_idx = None
                if class_idx is not None:
                    class_labels[:, :] = 0
                    class_labels[:, class_idx] = 1
                label_cllt.append(class_labels.cpu())
            if data_step < current_loaded_number:
                continue
            if 'gt_v' in config.sampling.direction:
                batch, nfe = sampling_fn(analytic_v, z0)
            else:
                batch, nfe = sampling_fn(score_model, z0, label=class_labels)
            batch = scaler(batch)    # [-1, 1]
            # print(batch.shape, batch.max(), batch.min(), z0.mean(), z0.std())
            z0_cllt.append(z0.cpu())
            data_cllt.append(batch.cpu())
            nfes.append(nfe)
            # save intermediate results
            if (data_step + 1) % 10 == 0:
                save_data_pair(seeded_data_root, z0_cllt, data_cllt, total_number_of_samples, class_cllt=label_cllt, z0_name='z0_tmp.npy', z1_name='z1_tmp.npy')

        save_data_pair(seeded_data_root, z0_cllt, data_cllt, total_number_of_samples, class_cllt=label_cllt)
        delete_tmp_data(seeded_data_root)
        logging.info(f'Successfully generated z1 from random z0 with random seed: {config.seed}, ave nfe: {np.mean(nfes):0.6f}')
        sys.exit(0)
    
    elif 'from_z1' in config.sampling.direction:
        if config.data.random_flip:
            logging.warning('random flip is enabled, please check if it is correct')
        
        logging.info(f'Start generating data with ODE from z1, SEED: {config.seed}')
        
        if FLAGS.z1_path:
            z1_loaded = np.load(FLAGS.z1_path)
            z1_loaded = torch.from_numpy(z1_loaded).to(config.device)
            logging.info(f'loaded z1 shape: {z1_loaded.shape} from {FLAGS.z1_path}')
            total_number_of_samples = z1_loaded.shape[0]
            num_iter = int(np.ceil(total_number_of_samples / config.training.batch_size))
            pbar = tqdm(range(num_iter))
        else:
            # dataloader
            dataloader = datasets.get_dataset(config)
            logger.info(f'length of dataloader: {len(dataloader)}')
            total_number_of_samples = config.sampling.total_number_of_samples
            num_iter = int(np.ceil(total_number_of_samples / config.training.batch_size))
            pbar = tqdm(range(num_iter))

        ema.copy_to(score_model.parameters())
        data_cllt = []
        z1_cllt = []
        label_cllt = []
        nfes = []
        # if z1_tmp and z0_tmp exist, load them
        if os.path.exists(os.path.join(seeded_data_root, 'z1_tmp.npy')) and os.path.exists(os.path.join(seeded_data_root, 'z0_tmp.npy')):
            data_tmp_loaded, z1_tmp_loaded = load_tmp_data(seeded_data_root)
            data_cllt.append(data_tmp_loaded)
            z1_cllt.append(z1_tmp_loaded)
            assert z1_tmp_loaded.shape[0] == data_tmp_loaded.shape[0]
            current_loaded_number = z1_tmp_loaded.shape[0] // config.training.batch_size
            logging.info(f'current data_tmp loaded shape: {data_tmp_loaded.shape}, current z1_tmp loaded shape: {z1_tmp_loaded.shape}')
        else:
            current_loaded_number = 0
            
        for data_step in pbar:
            if FLAGS.z1_path:
                z1 = z1_loaded[data_step*config.training.batch_size:(data_step+1)*config.training.batch_size]
                label_dic = torch.zeros(z1.shape[0]) # dummy label
            else:
                try:
                    z1, label_dic = next(data_iterator)
                except:
                    data_iterator = iter(dataloader)
                    z1, label_dic = next(data_iterator)
            if data_step < current_loaded_number:
                continue
            z1 = z1.to(config.device)
            if 'gt_v' in config.sampling.direction:
                ## traped in singularities
                del analytic_v
                analytic_v = AnalyticFlow(gt_images).to(device)
                batch, nfe = sampling_fn(analytic_v, z1, reverse=True)
            else:
                batch, nfe = sampling_fn(score_model, z1, reverse=True)
            batch = scaler(batch)    # [-1, 1]
            # print(batch.shape, batch.max(), batch.min(), z0.mean(), z0.std())
            z1_cllt.append(z1.cpu())
            data_cllt.append(batch.cpu())
            label_cllt.append(label_dic.cpu()) if (label_dic is not None) else None
            nfes.append(nfe)
            # save intermediate results
            if (data_step + 1) % 10 == 0:
                save_data_pair(seeded_data_root, data_cllt, z1_cllt, total_number_of_samples, z0_name='z0_tmp.npy', z1_name='z1_tmp.npy')
        save_data_pair(seeded_data_root, data_cllt, z1_cllt, total_number_of_samples)
        if len(label_cllt) > 0:
            label_cllt = torch.cat(label_cllt).cpu().numpy()
            np.save(os.path.join(seeded_data_root, 'label.npy'), label_cllt[:total_number_of_samples])
        delete_tmp_data(seeded_data_root)
        logging.info(f'Successfully generated z0 from training set z1 with random seed: {config.seed}, ave nfe: {np.mean(nfes):0.6f}')
        sys.exit(0)
         
    elif config.sampling.direction == 'random_paired':
        logging.info(f'create random paired data using training set, SEED: {config.seed}')
        # dataloader
        dataloader = datasets.get_dataset(config)
        logger.info(f'length of dataloader: {len(dataloader)}')

        data_cllt = []
        z1_cllt = []
        total_number_of_samples = config.sampling.total_number_of_samples
        num_iter = int(np.ceil(total_number_of_samples / config.training.batch_size))
        pbar = tqdm(range(num_iter))
        for data_step in pbar:
            try:
                z1, label_dic = next(data_iterator)
            except:
                data_iterator = iter(dataloader)
                z1, label_dic = next(data_iterator)
            batch = torch.randn_like(z1)
            z1_cllt.append(z1.cpu())
            data_cllt.append(batch.cpu())
            # save intermediate results
            if (data_step + 1) % 10 == 0:
                save_data_pair(seeded_data_root, data_cllt, z1_cllt, total_number_of_samples, z0_name='z0_tmp.npy', z1_name='z1_tmp.npy')
        save_data_pair(seeded_data_root, data_cllt, z1_cllt, total_number_of_samples)
        delete_tmp_data(seeded_data_root)
        logging.info(f'Successfully generated random paired z0 from training set z1 with random seed: {config.seed}')
        sys.exit(0)


if __name__ == "__main__":
    app.run(main)

