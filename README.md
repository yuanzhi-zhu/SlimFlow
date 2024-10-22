# SlimFlow

This is the official implementation of ECCV2024 paper
## [SlimFlow: Training Smaller One-Step Diffusion Models with Rectified Flow](https://arxiv.org/abs/2407.12718) 
by [Yuanzhi Zhu](https://yuanzhi-zhu.github.io/about/), [Xingcaho Liu](https://gnobitab.github.io/), [Qiang Liu](https://www.cs.utexas.edu/~lqiang/)

This code is based on [RectifiedFlow](https://github.com/gnobitab/RectifiedFlow).

## usage

### Train 1-Rectified Flow

```bash
python ./train.py \
    --config ./configs/rectified_flow/cifar10_rf_gaussian.py  \
    --config.expr 1_rectified_flow \
```

### Evaluation 

evaluate FID of ckpts from `config.eval.begin_ckpt` in `ckpt_dir`
#### one step
```bash
python ./evaluation_fid.py \
    --config ./configs/rectified_flow/cifar10_rf_gaussian.py  \
    --ckpt_dir logs/1_rectified_flow \
    --config.eval.batch_size 512 --config.eval.num_samples 50000 \
    --config.eval.begin_ckpt 1 --config.eval.end_ckpt 0 \
    --config.sampling.sample_N 1 --config.sampling.use_ode_sampler euler \
```

#### rk45 by default
```bash
python ./evaluation_fid.py \
    --config ./configs/rectified_flow/cifar10_rf_gaussian.py  \
    --ckpt_dir logs/1_rectified_flow \
    --config.eval.batch_size 512 --config.eval.num_samples 50000 \
    --config.eval.begin_ckpt 1 --config.eval.end_ckpt 0 \
```

### Image Sampling

sampling all ckpts in `sampling_dir`
```bash
python ./image_sampling.py \
    --config ./configs/rectified_flow/cifar10_rf_gaussian.py \
    --sampling_dir "logs/1_rectified_flow" \
    --config.eval.batch_size 64
```

##### Image Sampling Configurations
- Sample from 1flows: `--config.sampling.use_ode_sampler rk45`
- Sample from 2flows: [`--config.sampling.use_ode_sampler rk45`, `--config.sampling.use_ode_sampler heun` + `--config.sampling.sample_N 3`, `--config.sampling.use_ode_sampler euler` + `--config.sampling.sample_N 1`]
- Sample from distilled one-step models: `--config.sampling.use_ode_sampler euler` + `--config.sampling.sample_N 1`

##### Model Configurations
- ImageNet64 80.7M: `--config.model.name DhariwalUNet --config.model.nf 128 --config.model.num_res_blocks 2 --config.model.ch_mult '(1, 2, 2, 4)' --config.data.num_classes 1000 --config.data.image_size 64 --config.model.attn_resolutions '32, 16'`
- ImageNet 44.7MM: `--config.model.name DhariwalUNet --config.model.nf 128 --config.model.num_res_blocks 2 --config.model.ch_mult '(1, 2, 2, 2)' --config.data.num_classes 1000 --config.data.image_size 64 --config.model.attn_resolutions '32, 16'`
- FFHQ64 27.9M: `--config.model.nf 128 --config.model.num_res_blocks 2 --config.data.image_size 64 --config.model.ch_mult '(1, 2, 2)'`
- FFHQ64 15.7M: `--config.model.nf 96 --config.model.num_res_blocks 2 --config.data.image_size 64 --config.model.ch_mult '(1, 2, 2)'`
- FFHQ64 7.0M: `--config.model.nf 64 --config.model.num_res_blocks 2 --config.data.image_size 64 --config.model.ch_mult '(1, 2, 2)'`
- FFHQ64 3.4M: `--config.model.nf 64 --config.model.num_res_blocks 1 --config.data.image_size 64 --config.model.ch_mult '(1, 1, 2)'`
- CIFAR32 27.9M: `--config.model.nf 128 --config.model.num_res_blocks 2 --config.data.image_size 32 --config.model.ch_mult '(1, 2, 2)'`
- CIFAR32 15.7M: `--config.model.nf 96 --config.model.num_res_blocks 2 --config.data.image_size 32 --config.model.ch_mult '(1, 2, 2)'`
- CIFAR32 7.0M: `--config.model.nf 64 --config.model.num_res_blocks 2 --config.data.image_size 32 --config.model.ch_mult '(1, 2, 2)'`
- CIFAR32 3.4M: `--config.model.nf 64 --config.model.num_res_blocks 1 --config.data.image_size 32 --config.model.ch_mult '(1, 1, 2)'`

### Generate Data Pair

#### z0-->z1 by default
```bash
python ./generate_data.py \
    --config ./configs/rectified_flow/cifar10_rf_gaussian.py  \
    --ckpt_path "logs/1_rectified_flow/checkpoints/checkpoint_14.pth" \
    --data_root "reflow_data/1_rectified_flow_50000/" \
    --config.sampling.total_number_of_samples 50000 --config.seed 0 \
    --config.training.batch_size 512 \
    --config.sampling.direction from_z0 \
```

`config.sampling.direction` has 3 options: 'from_z0', 'from_z1', 'random_paired'


### Reflow to get 2-Rectified Flow with the Generated Data Pair

```bash
python ./train.py \
    --config ./configs/rectified_flow/cifar10_rf_gaussian.py  \
    --config.data.reflow_data_root "reflow_data/1_rectified_flow_50000/" \
    --config.flow.flow_t_schedule uniform \
    --config.expr 2_rectified_flow \
    --config.flow.h_flip=true \
    --config.flow.pre_train_model /logs/1_rectified_flow/checkpoints/checkpoint_14.pth \
```

### Annealing Reflow

```bash
python ./train.py \
    --config ./configs/rectified_flow/cifar10_rf_gaussian.py  \
    --config.expr 2_rectified_flow_500001flow_flip_warmup_300000_28m \
    --config.flow.h_flip=true \
    --config.training.x0_randomness warmup_300000 \
    --config.training.snapshot_freq 50000 \
    --config.training.snapshot_sampling 10000 \
    --config.data.reflow_data_root "reflow_data/1_rectified_flow_50000/" \
    --config.model.nf 128 --config.model.num_res_blocks 2 \
    --config.model.ch_mult '(1, 2, 2)' \
```

must specify `config.data.data_root` for reflow training

if `config.flow.pre_train_model` is not specified, the model will be trained from scratch.

### Distill to get one-step model
<!-- distillation as special case of reflow with different `flow_t_schedule` and `flow_alpha_t` -->

```bash
python ./train.py \
    --config ./configs/rectified_flow/cifar10_rf_gaussian.py  \
    --config.data.reflow_data_root "reflow_data/1_rectified_flow_50000/" \
    --config.flow.flow_t_schedule t0 \
    --config.training.loss_type lpips \
    --config.flow.use_teacher true \
    --config.expr 2_rectified_flow_500000bigflow_28m_distill_lpips_use_teacher \
    --config.flow.pre_train_model "./logs/2_rectified_flow_500001flow_flip_warmup_300000_28m/checkpoints/checkpoint_16.pth" \
    --config.model.nf 128 --config.model.num_res_blocks 2 \
    --config.model.ch_mult '(1, 2, 2)' \
```

## Checkpoints
checkpoints can be found here on HuggingFace: https://huggingface.co/Yuanzhi/SlimFlow
To sample from these checkpoints, please follow the instructions in the README.md of the HuggingFace model.

## Citation
If you find this repo helpful, please cite:

```bibtex
@article{zhu2024slimflow,
  title={SlimFlow: Training Smaller One-Step Diffusion Models with Rectified Flow},
  author={Zhu, Yuanzhi and Liu, Xingchao and Liu, Qiang},
  journal={arXiv preprint arXiv:2407.12718},
  year={2024}
}
```
