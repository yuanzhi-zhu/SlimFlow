from typing import Any, Iterable, Tuple, Union
import numpy as np
import torch
from torch import Tensor, nn
import logging
from models.ema import ExponentialMovingAverage
from reflow.losses import get_rectified_flow_loss_fn

class RectifiedFlow():
    def __init__(self, model=None, ema_model=None, cfg=None):
        self.cfg = cfg
        self.model = model
        ## init ema model
        if ema_model == None:
            # self.ema_model = copy.deepcopy(self.model)
            self.ema_model = ExponentialMovingAverage(self.model.parameters(), decay=self.cfg.model.ema_rate)
        else:
            self.ema_model = ema_model
        self.device = self.cfg.device
        
        if 'lpips' in self.cfg.training.loss_type:
            import lpips
            self.lpips_model = lpips.LPIPS(net='vgg')
            self.lpips_model = self.lpips_model.cuda()
            for p in self.lpips_model.parameters():
                p.requires_grad = False
        ## parameters
        self.eps = self.cfg.flow.eps
        self.use_ode_sampler = self.cfg.sampling.use_ode_sampler
        self.init_type = self.cfg.sampling.init_type
        self.sample_N = self.cfg.sampling.sample_N
        self.ode_tol = self.cfg.sampling.ode_tol
        self.noise_scale = self.cfg.sampling.init_noise_scale
        try:
            self.flow_t_schedule = int(self.cfg.flow.flow_t_schedule)
        except:
            self.flow_t_schedule = self.cfg.flow.flow_t_schedule
        self.flow_alpha_t = self.cfg.flow.flow_alpha_t
        ## get loss function
        self.loss_fn = get_rectified_flow_loss_fn(self.cfg.training.reduce_mean, self.cfg.training.loss_type)
        if self.cfg.flow.use_teacher:
            self.loss_fn_teach = get_rectified_flow_loss_fn(self.cfg.training.reduce_mean, self.cfg.training.loss_type)
        # Initialize the _T instance variable to a default value
        self._T = 1.
        ## x0 randomness
        if 'warmup' in self.cfg.training.x0_randomness:
            self.warmup_iters = int(self.cfg.training.x0_randomness.split('_')[-1])
            logging.info(f'x0_randomness warmup type: {self.cfg.training.x0_randomness}; warmup_iters: {self.warmup_iters}')
        else:
            logging.info(f'x0_randomness type: {self.cfg.training.x0_randomness}')

        logging.info(f'Init. Distribution Variance: {self.noise_scale}')
        logging.info(f'SDE Sampler Variance: 0 for flow')
        logging.info(f'ODE Tolerence: {self.ode_tol}')

    @property
    def T(self):
        return self._T

    @T.setter
    def T(self, value):
        self._T = value

    def lpips_forward_wrapper(self, x, y, size=64):
        # resize (B, C, H, W) to (B, C, 224, 224)
        if size > 0:
            x = nn.functional.interpolate(x, size=(size, size), mode='bilinear', align_corners=True)
            y = nn.functional.interpolate(y, size=(size, size), mode='bilinear', align_corners=True)
        return self.lpips_model(x, y) if 'lpips' in self.cfg.training.loss_type else None
        
    def randomness_schedule(self, current_training_step) -> float:
        """Randomness at the current point in training."""
        # if cfg.training.x0_randomness is a number, use it as fixed randomness
        if 'fix' in self.cfg.training.x0_randomness:
            self.x0_randomness = float(self.cfg.training.x0_randomness.split('_')[1])
        elif 'warmup' in self.cfg.training.x0_randomness:
            if 'exp' in self.cfg.training.x0_randomness:
                # use exponential warmup, from 1 to 0
                self.x0_randomness = np.exp(-current_training_step / self.warmup_iters)
            elif 'cos' in self.cfg.training.x0_randomness:
                # use cosine warmup, from 1 to 0
                self.x0_randomness = (1 + np.cos(np.pi * min(1, current_training_step / self.warmup_iters))) / 2
                # self.x0_randomness = np.cos(np.pi / 2 * min(1, current_training_step / self.warmup_iters))
            else:
                # use linear warmup, from 1 to 0
                self.x0_randomness = 1 - min(1, current_training_step / self.warmup_iters)
        else:
            raise NotImplementedError(f'x0_randomness {self.cfg.training.x0_randomness} Not implemented')

    def model_forward_wrapper(
        self,
        model: nn.Module,
        x: Tensor,
        t: Tensor,
        **kwargs: Any,
    ) -> Tensor:
        """Wrapper for the model call"""
        label = kwargs['label'] if 'label' in kwargs else None
        augment_labels = kwargs['augment_labels'] if 'augment_labels' in kwargs else None
        # x = torch.cat([x, self.cond], dim=1) if self.cfg.flow.use_cond else x
        # model_output = model(x, t*999, label, augment_labels)
        model_output = model(x, t*999) if label is None else model(x, t*999, label.float())
        model_output = model_output[0] if isinstance(model_output, tuple) else model_output
        return model_output
    
    def get_data_pair(self, batch):
        # construct data pair
        if self.cfg.data.reflow_data_root:    # paired data from flow
            # split batch into clean data and noise with equal batch size
            self.data, z0 = torch.split(batch, 1, dim=1)
            self.data = self.data.squeeze(1)
            self.noise = z0.squeeze(1)
            # add perturbation to z0
            if self.x0_randomness > 0:
                self.noise = np.sqrt(1 - self.x0_randomness**2) * self.noise + self.x0_randomness * torch.randn_like(self.noise)
            if self.cfg.flow.h_flip:
                # flip the data with 50% probability when x0_randomness == 1
                if np.random.rand() > 0.5:
                    self.data = torch.flip(self.data, dims=[3])
                    self.noise = torch.flip(self.noise, dims=[3])
        else:   # unpaired data
            self.data = batch
            self.noise = torch.randn_like(self.data)

    def get_interpolations(self,
                        data: Tensor,
                        noise: Tensor,
                        **kwargs: Any,):
        """get t, x_t based on the flow time schedule and alpha schedule"""
        # sample timesteps
        if self.flow_t_schedule == 't0': ### distill for t = 0 (k=1)
            self.t = torch.zeros((data.shape[0],), device=data.device) * (self.T - self.eps) + self.eps
        elif self.flow_t_schedule == 't1': ### reverse distill for t=1 (fast embedding)
            self.t = torch.ones((data.shape[0],), device=data.device) * (self.T - self.eps) + self.eps
        elif self.flow_t_schedule == 't0t1': ### t = 0, 1, two ends of the trajectory
            self.t = torch.randint(0, 2, (data.shape[0],), device=data.device) * (self.T - self.eps) + self.eps
        elif self.flow_t_schedule == 'uniform': ### train new rectified flow with reflow
            self.t = torch.rand((data.shape[0],), device=data.device) * (self.T - self.eps) + self.eps
        elif 'fix' in self.flow_t_schedule: ### train new rectified flow with fixed t
            t = float(self.flow_t_schedule.split('_')[1])
            self.t = t * torch.ones((data.shape[0],), device=data.device) * (self.T - self.eps) + self.eps
        elif 'range' in self.flow_t_schedule: ### train new rectified flow with fixed t
            t_min, t_max = float(self.flow_t_schedule.split('_')[1]), float(self.flow_t_schedule.split('_')[2])
            self.t = torch.rand((data.shape[0],), device=data.device) * (t_max - t_min) + t_min
        elif type(self.flow_t_schedule) == int: ### k > 1 distillation
            self.t = torch.randint(0, self.flow_t_schedule, (data.shape[0],), device=data.device) * (self.T - self.eps) / self.flow_t_schedule + self.eps
        else:
            assert False, f'flow_t_schedule {self.flow_t_schedule} Not implemented'
        # mapping to alpha_t
        if self.flow_alpha_t == 't0': ### distill for t = 0 (k=1)
            self.alpha_t = torch.zeros((data.shape[0],), device=data.device) * (self.T - self.eps) + self.eps
        elif self.flow_alpha_t == 't1': ### reverse distill for t=1 (fast embedding)
            self.alpha_t = torch.ones((data.shape[0],), device=data.device) * (self.T - self.eps) + self.eps
        elif self.flow_alpha_t == 'uniform': ### train new rectified flow with reflow
            self.alpha_t = self.t
        else:
            assert False, f'flow_alpha_t {self.flow_alpha_t} Not implemented'
        # linear interpolation between clean image and noise
        self.xt = torch.einsum('b,bijk->bijk', self.alpha_t, data) + torch.einsum('b,bijk->bijk', (1 - self.alpha_t), noise)

    def pred_batch_outputs(self, **kwargs: Any,) -> Tuple[Tensor, Tensor]:
        """ Get the predicted and target values for computing the loss. """
        # get prediction with score model
        predicted = self.model_forward_wrapper(
            self.model,
            self.xt,
            self.t,
            **kwargs,
        )
        target = self.data - self.noise

        return predicted, target

    def teacher_loss(self, **kwargs: Any,):
        """ Get the predicted and target values for computing the loss. """
        noise = torch.randn_like(self.data)
        predicted = self.model_forward_wrapper(
            self.model,
            noise,
            self.eps * torch.ones((noise.shape[0],), device=noise.device),
            **kwargs,
        )
        with torch.no_grad():
            predicted_teacher_0 = self.model_forward_wrapper(
                self.model_teacher,
                noise,
                self.eps * torch.ones((noise.shape[0],), device=noise.device),
                **kwargs,
            )
            # t_ = torch.rand((predicted.shape[0],), device=predicted.device) * (self.T - self.eps) + self.eps
            t_ = torch.rand((predicted.shape[0],), device=predicted.device) * 0.6 + 0.2     # 0.2 ~ 0.8
            x_t_psuedo = noise + torch.einsum('b,bijk->bijk', t_, predicted_teacher_0)
            # get prediction with score model
            predicted_teacher_t = self.model_forward_wrapper(
                self.model_teacher,
                x_t_psuedo,
                t_,
                **kwargs,
            )
            x_1 = x_t_psuedo + torch.einsum('b,bijk->bijk', (1-t_), predicted_teacher_t)
            predicted_teacher = x_1 - noise
        loss = self.loss_fn_teach(self, predicted, predicted_teacher.detach(), noise=noise, t=t_)
        return loss

    def teacher_loss_new(self, **kwargs: Any,):
        """ Get the predicted and target values for computing the loss. """
        noise = torch.randn_like(self.data)
        predicted = self.model_forward_wrapper(
            self.model,
            noise,
            self.eps * torch.ones((noise.shape[0],), device=noise.device),
            **kwargs,
        )
        with torch.no_grad():
            # t_ = torch.rand((predicted.shape[0],), device=predicted.device) * (self.T - self.eps) + self.eps
            t_ = torch.rand((predicted.shape[0],), device=predicted.device) * 0.6 + 0.2     # 0.2 ~ 0.8
            x_t_psuedo = noise + torch.einsum('b,bijk->bijk', t_, predicted)
            # get prediction with score model
            predicted_teacher = self.model_forward_wrapper(
                self.model_teacher,
                x_t_psuedo,
                t_,
                **kwargs,
            )
            # can comment out the next two lines
            x_1 = x_t_psuedo + torch.einsum('b,bijk->bijk', (1-t_), predicted_teacher)
            predicted_teacher = x_1 - noise
        loss = self.loss_fn_teach(self, predicted, predicted_teacher.detach(), noise=noise, t=t_)
        return loss

    def train_step(self, batch, current_training_step: int, augment_pipe=None, **kwargs: Any,):
        """Performs a training step"""
        ### get loss
        '''
        batch: Clean data.
        current_training_step: global training step
        '''
        self.randomness_schedule(current_training_step)
        ## augment pipeline: edm --> https://github.com/NVlabs/edm/blob/main/training/augment.py
        batch, augment_labels = augment_pipe(batch) if augment_pipe is not None else (batch, None)
        # kwargs['augment_labels'] = augment_labels
        ## get data pair (self.data, self.noise)
        self.get_data_pair(batch)
        ## get interpolation t, x_t
        self.get_interpolations(self.data, self.noise)
        ## get prediction and target
        predicted, target = self.pred_batch_outputs(**kwargs)
        ## calculate loss
        loss = self.loss_fn(self, predicted, target)
        if self.cfg.flow.use_teacher:
            loss_teacher = self.teacher_loss_new(**kwargs)
            loss = loss + loss_teacher
        return loss

    def get_z0(self, batch, train=True):
        n,c,h,w = batch.shape 
        if self.init_type == 'gaussian':
            ### standard gaussian #+ 0.5
            cur_shape = (n, c, h, w)
            return torch.randn(cur_shape) * self.noise_scale
        else:
            raise NotImplementedError("INITIALIZATION TYPE NOT IMPLEMENTED") 


### Analytic Teacher Model ###
### equivalent velocity model ###
class AnalyticFlow(nn.Module):
    def __init__(self, gt_images):
        super().__init__()
        self.gt_images = gt_images
        self.gt_data = self.gt_images.clone().view(self.gt_images.shape[0], -1).detach()
        self.t_schedule = 999
        self.linear_weight = -2. * self.gt_data.T
        self.linear_bias = self.gt_data.norm(dim=-1, keepdim=False).pow(2)

    def forward(self, xt, t):
        ## \text{Softmax}\left( \frac{-X_t^T X_t + 2t(X_1^{(i)})^TX_t - t^2\|X_1\|_2^2}{2(1 - t)^2} \right)  \frac{X_1^{(i)}}{1-t}
        t = t / self.t_schedule     # reduce t to [0, 1], in shape (batch_size, )
        x_flat = xt.view(xt.shape[0], -1)
        quad = torch.einsum('bi,bi->b', x_flat, x_flat).unsqueeze(-1)
        linear = torch.einsum('b,bi,ik->bk', t, x_flat, self.linear_weight)
        bias = torch.einsum('b,bk->bk', t**2, self.linear_bias.unsqueeze(0).repeat(xt.shape[0], 1))
        log_prob = - torch.einsum('b,bk->bk', 1 / (2.*(1. - t)**2), quad + linear + bias)
        ### NOTE: original softmax
        logit = torch.softmax(log_prob, dim=-1)
        weighted_sum = torch.einsum('bp,pchw->bchw', logit, self.gt_images)
        output = torch.einsum('b,bchw->bchw', 1 / (1. - t), weighted_sum - xt)
        return output

# ### Kernel Density Estimation with bandwidth h ###
# class KDEFlow(nn.Module):
#     def __init__(self, gt_images, h):
#         super().__init__()
#         self.gt_images = gt_images
#         self.gt_data = self.gt_images.clone().view(self.gt_images.shape[0], -1).detach()
#         self.t_schedule = 999
#         self.linear_weight = -2. * self.gt_data.T
#         self.linear_bias = self.gt_data.norm(dim=-1, keepdim=False).pow(2)
#         self.h = h

#     def forward(self, xt, t):
#         ## \text{Softmax}\left( \frac{-X_t^T X_t + 2t(X_1^{(i)})^TX_t - t^2\|X_1\|_2^2}{2(1 - t)^2} \right)  \frac{X_1^{(i)}}{1-t}
#         t = t / self.t_schedule     # reduce t to [0, 1], in shape (batch_size, )
#         x_flat = xt.view(xt.shape[0], -1)
#         quad = torch.einsum('bi,bi->b', x_flat, x_flat).unsqueeze(-1)
#         linear = torch.einsum('b,bi,ik->bk', t, x_flat, self.linear_weight)
#         bias = torch.einsum('b,bk->bk', t**2, self.linear_bias.unsqueeze(0).repeat(xt.shape[0], 1))
#         log_prob = - torch.einsum('b,bk->bk', 1 / (2.*self.h**2), quad + linear + bias)
#         ### NOTE: original softmax
#         logit = torch.softmax(log_prob, dim=-1)
#         weighted_sum = torch.einsum('bp,pchw->bchw', logit, self.gt_images)
#         output = torch.einsum('b,bchw->bchw', 1 / (1. - t), weighted_sum - xt)
#         return output
