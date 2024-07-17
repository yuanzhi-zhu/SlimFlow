import os
import time
# import sys
join = os.path.join
from absl import app
from absl import flags
from ml_collections.config_flags import config_flags
import logging
import torch
import torchvision
import numpy as np
from models import utils as mutils
from models.ema import ExponentialMovingAverage
# Keep the import below for registering all model definitions
from models import ddpm, ncsnv2, ncsnpp
import reflow.datasets as datasets
from reflow.utils import restore_checkpoint, seed_everywhere
from reflow import RectifiedFlow
from reflow import losses as losses
from reflow import sampling as sampling

import tqdm
from cleanfid import fid

extensions = ['*.jpg', '*.jpeg', '*.JPEG', '*.png', '*.bmp']

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
  "config", None, "Training configuration.", lock_config=True)
flags.DEFINE_string("ckpt_dir", None, "checkpoint path.")
flags.DEFINE_string("data_dir", None, "checkpoint path.")
flags.DEFINE_string("target_path", None, "target path.")
flags.DEFINE_boolean("calc_flops", True, "calculate flops and macs.")
flags.DEFINE_boolean("use_ema", True, "Whether to use ema.")
flags.mark_flags_as_required(["config"])

class ResizeDataset(torch.utils.data.Dataset):
    """
    A placeholder Dataset that enables parallelizing the resize operation
    using multiple CPU cores

    files: list of all files in the folder
    fn_resize: function that takes an np_array as input [0,255]
    """

    def __init__(self, files, mode, size=(299, 299), fdir=None):
        self.files = files
        self.fdir = fdir
        self.transforms = torchvision.transforms.ToTensor()
        self.size = size
        self.fn_resize = fid.build_resizer(mode)
        self.custom_image_tranform = lambda x: x

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img_np = self.files[i]
        # apply a custom image transform before resizing the image to 299x299
        img_np = self.custom_image_tranform(img_np)
        # fn_resize expects a np array and returns a np array
        img_resized = self.fn_resize(img_np)

        # ToTensor() converts to [0,1] only if input in uint8
        if img_resized.dtype == "uint8":
            img_t = self.transforms(np.array(img_resized)) * 255
        elif img_resized.dtype == "float32":
            img_t = self.transforms(img_resized)

        return img_t


# https://github.com/openai/consistency_models_cifar10/blob/main/jcm/metrics.py#L117
def compute_fid(
    samples,
    feat_model,
    dataset_name="cifar10",
    ref_stat=None,
    dataset_res=32,
    dataset_split="train",
    batch_size=512,
    num_workers=12,
    mode="legacy_tensorflow",
    device=torch.device("cuda:0"),
    seed=0,
):
    dataset = ResizeDataset(samples, mode=mode)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )
    l_feats = []
    for batch in tqdm.tqdm(dataloader):
        l_feats.append(fid.get_batch_features(batch, feat_model, device))
    np_feats = np.concatenate(l_feats)
    mu = np.mean(np_feats, axis=0)
    sigma = np.cov(np_feats, rowvar=False)
    if ref_stat is not None:
        ref_mu, ref_sigma = ref_stat
    else:
        ref_mu, ref_sigma = fid.get_reference_statistics(
            dataset_name, dataset_res, mode=mode, seed=seed, split=dataset_split
        )
    score = fid.frechet_distance(mu, sigma, ref_mu, ref_sigma)

    return score


def main(argv):
    config = FLAGS.config
    assert (FLAGS.ckpt_dir is not None) or (FLAGS.data_dir is not None), "ckpt_dir or data_dir must be specified."
    
    ### basic info
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config.device = device
    logging.info(f'Using device: {device}; version: {str(torch.version.cuda)}')

    ### build feature extractor
    mode = "legacy_tensorflow"
    feat_model = fid.build_feature_extractor(mode, device)
    
    ### set random seed everywhere
    seed_everywhere(config.seed)

    # get checkpoint list
    workdir = FLAGS.ckpt_dir if FLAGS.ckpt_dir is not None else os.path.dirname(FLAGS.data_dir)
    checkpoint_dir = os.path.join(workdir, "checkpoints")
    
    # set up logger
    gfile_stream = open(f'{workdir}/eval_stdout.log', 'a+')
    handler = logging.StreamHandler(gfile_stream)
    formatter = logging.Formatter('%(filename)s - %(asctime)s - %(levelname)s --> %(message)s')
    handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler)
    logger.setLevel('INFO')

    ### load target samples amd calculate reference statistics
    if FLAGS.target_path:
        logging.info(f'load target samples from {FLAGS.target_path}')
        try:
            ## from https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/
            ref = np.load(FLAGS.target_path)
            ref_mu, ref_sigma = ref['mu'], ref['sigma']
            ref_stat = (ref_mu, ref_sigma)
            logging.info(f'reference statistics loaded!')
        except:
            target_samples = np.load(FLAGS.target_path)
            target_samples = torch.from_numpy(target_samples)
            target_samples = target_samples / 2 + 0.5
            target_samples = np.clip(target_samples.permute(0, 2, 3, 1).cpu().numpy() * 255., 0, 255).astype(np.uint8)
            target_samples = target_samples.reshape(
            (-1, config.data.image_size, config.data.image_size, config.data.num_channels))
            target_dataset = ResizeDataset(target_samples, mode=mode)
            target_dataloader = torch.utils.data.DataLoader(
                target_dataset,
                batch_size=512,
                shuffle=False,
                drop_last=False,
                num_workers=0,
            )
            l_feats = []
            for batch in tqdm.tqdm(target_dataloader):
                l_feats.append(fid.get_batch_features(batch, feat_model, device))
            np_feats = np.concatenate(l_feats)
            ref_mu = np.mean(np_feats, axis=0)
            ref_sigma = np.cov(np_feats, rowvar=False)
            ref_stat = (ref_mu, ref_sigma)
            logging.info(f'reference statistics calcualted!')
    else:
        ref_stat = None

    ### calculate fid for given data
    if FLAGS.data_dir is not None:
        logging.info(f'calculate fid for data from {FLAGS.data_dir}')
        samples = np.load(FLAGS.data_dir)
        logging.info(f'samples shape: {samples.shape}')
        logging.info(f'samples range: {samples.min()}, {samples.max()}')
        samples = torch.from_numpy(samples)
        samples = samples / 2 + 0.5
        samples = np.clip(samples.permute(0, 2, 3, 1).cpu().numpy() * 255., 0, 255).astype(np.uint8)
        all_samples = samples.reshape(
        (-1, config.data.image_size, config.data.image_size, config.data.num_channels))
        fid_score = compute_fid(
                    all_samples[: config.eval.num_samples],
                    mode=mode,
                    device=device,
                    feat_model=feat_model,
                    seed=config.seed,
                    num_workers=0,
                    ref_stat=ref_stat,
                )
        logging.info(f"data_dir-{FLAGS.data_dir} --- FID: {fid_score:0.6f}")
        return

    ### create model & optimizer
    # Initialize model.
    score_model = mutils.create_model(config) if config.model.name != 'DhariwalUNet' else mutils.create_model_edm(config)
    score_model.to(device)
    if FLAGS.calc_flops:
        from calflops import calculate_flops
        import torch.nn as nn
        # model wraper
        class Wrapper(nn.Module):
            def __init__(self, model):
                super(Wrapper, self).__init__()
                self.model = model
            def forward(self, x):
                t = torch.ones(x.shape[0], device=device)
                if config.data.num_classes:
                    class_labels = torch.eye(1000, device=device)[torch.randint(0, 1000, (x.shape[0],))]
                    return self.model(x, t*999, class_labels)
                return self.model(x, t*999)
        pratial_model = Wrapper(score_model)
        input_shape = (1, config.data.num_channels, config.data.image_size, config.data.image_size)
        flops, macs, params = calculate_flops(model=pratial_model, 
                                              input_shape=input_shape, 
                                              output_as_string=True, 
                                              output_precision=8,
                                              print_results=False)
        logging.info(f'FLOPs: {flops}; MACs: {macs}; Params: {params}')

    ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
    optimizer = losses.get_optimizer(config, score_model.parameters())
    state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)
    
    flow = RectifiedFlow(model=score_model, ema_model=ema, cfg=config)
    flow.model.eval()

    # Create data normalizer and its inverse
    scaler = datasets.get_data_scaler(config)
    inverse_scaler = datasets.get_data_inverse_scaler(config)

    # Building sampling functions
    sampling_shape = (config.eval.batch_size, config.data.num_channels,
                    config.data.image_size, config.data.image_size)
    sampling_fn = sampling.get_flow_sampler(flow, sampling_shape, inverse_scaler, device=device)

    logging.info(f'num of samples to evaluate: {config.eval.num_samples}')
    begin_ckpt = config.eval.begin_ckpt
    logging.info("begin checkpoint: %d" % (begin_ckpt,))
    # number of checkpoints in checkpoint_dir
    num_ckpts = len([name for name in os.listdir(checkpoint_dir) \
                     if (os.path.isfile(os.path.join(checkpoint_dir, name)) and 'checkpoint' in name)])
    end_ckpt = config.eval.end_ckpt if config.eval.end_ckpt > 0 else num_ckpts
    ckpt = begin_ckpt
    while ckpt <= end_ckpt:
        # reset random seed for each checkpoint
        seed_everywhere(config.seed)
        # Wait if the target checkpoint doesn't exist yet
        waiting_message_printed = False
        ckpt_filename = os.path.join(checkpoint_dir, "checkpoint_{}.pth".format(ckpt))
        while not os.path.exists(ckpt_filename):
            if not waiting_message_printed:
                logging.warning("Waiting for the arrival of checkpoint_%d" % (ckpt,))
                waiting_message_printed = True
            time.sleep(60)

        # Wait for 2 additional mins in case the file exists but is not ready for reading
        ckpt_path = os.path.join(checkpoint_dir, f'checkpoint_{ckpt}.pth')
        try:
            state = restore_checkpoint(ckpt_path, state, device=config.device)
            logging.info(f'load model from {ckpt_path}')
        except:
            time.sleep(60)
            try:
                state = restore_checkpoint(ckpt_path, state, device=config.device)
            except:
                time.sleep(120)
                state = restore_checkpoint(ckpt_path, state, device=config.device)
        if FLAGS.use_ema:
            logging.info("Using EMA for evaluation.")
            ema.copy_to(score_model.parameters())
        else:
            logging.info("Using non-EMA model for evaluation.")
        
        # sampling
        all_samples = []
        nfes = []
        num_sampling_rounds = config.eval.num_samples // config.eval.batch_size + 1
        for r in tqdm.tqdm(range(num_sampling_rounds)):
            z0 = flow.get_z0(torch.zeros(sampling_shape, device=device), train=False).to(device)
            class_labels = None
            if config.data.num_classes:
                class_labels = torch.eye(config.data.num_classes, device=device)[torch.randint(0, config.data.num_classes, (config.eval.batch_size,))]
            class_idx = None
            if class_idx is not None:
                class_labels[:, :] = 0
                class_labels[:, class_idx] = 1
            samples, nfe = sampling_fn(score_model, z=z0, label=class_labels, rho=config.sampling.rho)
            # logging.info("sampling -- ckpt: %d, round: %d, n: %d" % (ckpt, r, n))
            samples = np.clip(samples.permute(0, 2, 3, 1).cpu().numpy() * 255., 0, 255).astype(np.uint8)
            samples = samples.reshape(
            (-1, config.data.image_size, config.data.image_size, config.data.num_channels))
            all_samples.append(samples)
            nfes.append(nfe)

        # compute FID
        all_samples = np.concatenate(all_samples, axis=0)
        fid_score = compute_fid(
                    all_samples[: config.eval.num_samples],
                    mode=mode,
                    device=device,
                    feat_model=feat_model,
                    seed=config.seed,
                    num_workers=0,
                    ref_stat=ref_stat,
                )
        logging.info(f"ckpt-{ckpt} --- FID: {fid_score:0.6f}; avg nfe: {np.mean(nfes):0.6f}")
        # update the number of checkpoints in checkpoint_dir
        if ckpt == end_ckpt and config.eval.end_ckpt <= 0:
            end_ckpt = len([name for name in os.listdir(checkpoint_dir) \
                            if (os.path.isfile(os.path.join(checkpoint_dir, name)) and 'checkpoint' in name)])
        # move to the next checkpoint
        ckpt += 1


if __name__ == "__main__":
    app.run(main)
