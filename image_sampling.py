import os
# import sys
join = os.path.join
from absl import app
from absl import flags
import re
from ml_collections.config_flags import config_flags
import torch
from torchvision.utils import save_image
from models import utils as mutils
from models.ema import ExponentialMovingAverage
# Keep the import below for registering all model definitions
from models import ddpm, ncsnv2, ncsnpp
import reflow.datasets as datasets
from reflow.utils import restore_checkpoint, seed_everywhere
from reflow import RectifiedFlow
from reflow import losses as losses
from reflow import sampling as sampling
import matplotlib.pyplot as plt

extensions = ['*.jpg', '*.jpeg', '*.JPEG', '*.png', '*.bmp']

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
  "config", None, "Training configuration.", lock_config=True)
flags.DEFINE_string("sampling_dir", None, "Work directory.")
flags.mark_flags_as_required(["config", "sampling_dir"])


def main(argv):
    config = FLAGS.config

    ### basic info
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config.device = device
    print(f'Using device: {device}; version: {str(torch.version.cuda)}')

    ### set random seed everywhere
    seed_everywhere(config.seed)

    ### create model & optimizer
    # Initialize model.
    score_model = mutils.create_model(config) if config.model.name != 'DhariwalUNet' else mutils.create_model_edm(config)
    score_model.to(device)

    ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
    optimizer = losses.get_optimizer(config, score_model.parameters())
    state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

    # Load pre-trained model if specified
    flow = RectifiedFlow(model=score_model, ema_model=ema, cfg=config)
    flow.model.eval()

    # Create data normalizer and its inverse
    scaler = datasets.get_data_scaler(config)
    inverse_scaler = datasets.get_data_inverse_scaler(config)

    # get sampling_shape
    sampling_shape = (config.eval.batch_size, config.data.num_channels,
                    config.data.image_size, config.data.image_size)
    sampling_shape_ = (1, config.data.num_channels,
                    config.data.image_size, config.data.image_size)

    # reset random seed for Initial noise
    seed_everywhere(config.seed)
    # Initial noise fixed
    z0 = flow.get_z0(torch.zeros(sampling_shape, device=device), train=False).to(device)
    z = z0.detach().clone()
    z0_ = flow.get_z0(torch.zeros(sampling_shape_, device=device), train=False).to(device)
    z_ = z0_.detach().clone()
    class_labels = class_labels_ = None
    if config.data.num_classes:
        class_labels = torch.eye(config.data.num_classes, device=device)[torch.randint(0, config.data.num_classes, (config.eval.batch_size,))]
        class_idx = config.sampling.class_label
        if class_idx is not None and class_idx >= 0:
            class_labels[:, :] = 0
            class_labels[:, class_idx] = 1
        class_labels_ = torch.eye(config.data.num_classes, device=device)[torch.randint(0, config.data.num_classes, (1,))]

    workdir = FLAGS.sampling_dir
    if workdir.endswith('.pth'):
        # ckpt_list = [workdir]
        checkpoint_dir = os.path.dirname(workdir)
        ckpt_list = [os.path.basename(workdir)]
        begin_ckpt = 1
        end_ckpt = 2
        workdir = os.path.dirname(workdir)
    else:
        checkpoint_dir = os.path.join(workdir, "checkpoints")
        ckpt_list = os.listdir(checkpoint_dir)
        ckpt_list = [ckpt for ckpt in ckpt_list if ckpt.endswith(".pth")]
        ckpt_list = sorted(ckpt_list, key=lambda x: int(re.findall(r'\d+', x)[0])) # sort by number
        
        begin_ckpt = config.eval.begin_ckpt
        # number of checkpoints in checkpoint_dir
        num_ckpts = len([name for name in os.listdir(checkpoint_dir) \
                        if (os.path.isfile(os.path.join(checkpoint_dir, name)) and 'checkpoint' in name)])
        end_ckpt = config.eval.end_ckpt if config.eval.end_ckpt > 0 else num_ckpts
        print("sample from: ", ckpt_list, 'begin_ckpt:', begin_ckpt, 'end_ckpt:', end_ckpt)

    for ckpt in ckpt_list[begin_ckpt-1:end_ckpt]:
        # reset random seed for each checkpoint
        seed_everywhere(config.seed)
        
        eval_folder=f"eval_sample/{ckpt}"

        # Create directory to eval_folder
        eval_dir = os.path.join(workdir, eval_folder)
        os.makedirs(eval_dir, exist_ok=True)

        ckpt_path = os.path.join(checkpoint_dir, ckpt)
        
        # Load checkpoint
        state = restore_checkpoint(ckpt_path, state, device=config.device)
        print(f"load model from {ckpt_path}")
        ema.copy_to(score_model.parameters())

        # Setup ode solver: rk45
        flow.use_ode_sampler = 'rk45'
        sampling_fn = sampling.get_flow_sampler(flow, sampling_shape, inverse_scaler, clip_denoised=True, device=device)
        with torch.no_grad():
            x, nfe = sampling_fn(score_model, z=z, label=class_labels)
        save_image(x, os.path.join(eval_dir, f'{flow.use_ode_sampler}_sample_{nfe}.png'), nrow=8)

        # Setup ode solver: euler
        flow.use_ode_sampler = 'euler'
        for N in [1, 2, 5, 10, 20, 50, 100]:
            flow.sample_N = N
            sampling_fn = sampling.get_flow_sampler(flow, sampling_shape, inverse_scaler, clip_denoised=True, device=device)
            with torch.no_grad():
                x, nfe = sampling_fn(score_model, z=z, label=class_labels)
            save_image(x, os.path.join(eval_dir, f'{flow.use_ode_sampler}_sample_{nfe}.png'), nrow=8)

        ## calculate straightness
        for N in [100]:
            flow.sample_N = N
            dt = 1. / N
            sampling_fn = sampling.get_flow_sampler(flow, sampling_shape, inverse_scaler, clip_denoised=False, device=device)
            with torch.no_grad():
                x, nfe, (x_h, t_h) = sampling_fn(score_model, z=z, return_xh=True, progress=True, label=class_labels)
            x = scaler(x)
            v_final = (x - z).cpu() # [-1, 1]
            straightness = []
            for i in range(N):
                v_curr = (x_h[i+1] - x_h[i]) / dt
                # straight = torch.square(v_curr - v_final).view(v_curr.shape[0], -1).sum(dim=1)
                diff = (v_curr - v_final).view(v_curr.shape[0], -1)
                straight = torch.norm(diff, p='fro', dim=(1), keepdim=False)
                straightness.append(straight.mean() * dt)
            straightness = torch.stack(straightness)
            final_straightness = straightness.sum()
            print(f"straightness: {final_straightness}")

        ## Plot pixel trajectories
        num_pixel = 10
        for N in [100]:
            flow.sample_N = N
            sampling_fn = sampling.get_flow_sampler(flow, sampling_shape_, inverse_scaler, device=device)

            with torch.no_grad():
                x, nfe, (x_h, t_h) = sampling_fn(score_model, z=z_, return_xh=True, label=class_labels_)
            
            # Randomly sample n pixel positions
            h_indices = torch.randint(0, x.shape[2], (num_pixel,))
            w_indices = torch.randint(0, x.shape[3], (num_pixel,))

            pixels_h = [x[0, 0, h_indices, w_indices].detach().cpu() for x in x_h]

            plt.figure(figsize=(10, 5))
            # For each position in the tensor, plot a curve
            for position in range(pixels_h[0].shape[0]):
                plt.plot(t_h, [tensor[position] for tensor in pixels_h], label=f'pixel {position}')

            plt.title("pixel trajectories")
            plt.xlabel("time")
            plt.ylabel("pixel value")
            # plt.legend(loc="best")
            plt.grid(True)
            plt.savefig(os.path.join(eval_dir, f'{flow.use_ode_sampler}_pixel_traj_{nfe}.png'), dpi=600, bbox_inches='tight')  # dpi determines resolution, bbox_inches ensures the entire plot is saved
            plt.show()
            # Close the current figure
            plt.close()

if __name__ == "__main__":
    app.run(main)