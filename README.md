# SlimFlow: Training Smaller One-Step Diffusion Models with Rectified Flow

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

