import os
from tqdm import trange
import torch
from torch.nn import functional as F
from torch import distributions as dist
from dmifnet.common import (
    compute_iou, make_3d_grid
)
from dmifnet.utils import visualize as vis
from dmifnet.training import BaseTrainer


class Trainer(BaseTrainer):
    ''' Trainer object for the Occupancy Network.

    Args:
        model (nn.Module): Occupancy Network model
        optimizer (optimizer): pytorch optimizer object
        device (device): pytorch device
        input_type (str): input type
        vis_dir (str): visualization directory
        threshold (float): threshold value
        eval_sample (bool): whether to evaluate samples

    '''

    def __init__(self, model, optimizer, device=None, input_type='img',
                 vis_dir=None, threshold=0.5, eval_sample=False):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.input_type = input_type
        self.vis_dir = vis_dir
        self.threshold = threshold
        self.eval_sample = eval_sample

        if vis_dir is not None and not os.path.exists(vis_dir):
            os.makedirs(vis_dir)

    def train_step(self, data):
        ''' Performs a training step.

        Args:
            data (dict): data dictionary
        '''
        self.model.train()
        self.optimizer.zero_grad()
        loss, loss_ori, loss_head1, loss_head2, loss_head3, loss_head4 = self.compute_loss(data)
        loss.backward()
        self.optimizer.step()
        return loss.item(), loss_ori.item(), loss_head1.item(), loss_head2.item(), loss_head3.item(), loss_head4.item()

    def eval_step(self, data):
        ''' Performs an evaluation step.

        Args:
            data (dict): data dictionary
        '''
        self.model.eval()

        device = self.device
        threshold = self.threshold
        eval_dict = {}

        # Compute elbo
        points = data.get('points').to(device)
        occ = data.get('points.occ').to(device)

        inputs = data.get('inputs', torch.empty(points.size(0), 0)).to(device)
        voxels_occ = data.get('voxels')

        points_iou = data.get('points_iou').to(device)
        occ_iou = data.get('points_iou.occ').to(device)

        kwargs = {}

        with torch.no_grad():
            elbo, rec_error, kl = self.model.compute_elbo(
                points, occ, inputs, **kwargs)

        eval_dict['loss'] = -elbo.mean().item()
        eval_dict['rec_error'] = rec_error.mean().item()
        eval_dict['kl'] = kl.mean().item()

        # Compute iou
        batch_size = points.size(0)

        with torch.no_grad():
            p_out0, p_out1, p_out2, p_out3, p_out4 = self.model(points_iou, inputs,
                                                                sample=self.eval_sample, **kwargs)

        occ_iou_np = (occ_iou >= 0.5).cpu().numpy()

        # print(p_out)ssssss
        # print('th is', threshold)
        # threshold = threshold + 0.001 * (it - ori_it)
        occ_iou_hat_np = (p_out4.probs >= threshold).cpu().numpy()
        # print('th is', threshold)
        iou = compute_iou(occ_iou_np, occ_iou_hat_np).mean()
        eval_dict['iou'] = iou
        eval_dict['th'] = threshold

        # Estimate voxel iou
        if voxels_occ is not None:
            voxels_occ = voxels_occ.to(device)
            points_voxels = make_3d_grid(
                (-0.5 + 1 / 64,) * 3, (0.5 - 1 / 64,) * 3, (32,) * 3)
            points_voxels = points_voxels.expand(
                batch_size, *points_voxels.size())
            points_voxels = points_voxels.to(device)
            with torch.no_grad():
                p_out0, p_out1, p_out2, p_out3, p_out4 = self.model(points_voxels, inputs,
                                                                    sample=self.eval_sample, **kwargs)

            voxels_occ_np = (voxels_occ >= 0.5).cpu().numpy()
            occ_hat_np = (p_out4.probs >= threshold).cpu().numpy()
            iou_voxels = compute_iou(voxels_occ_np, occ_hat_np).mean()

            eval_dict['iou_voxels'] = iou_voxels

        return eval_dict

    def visualize(self, data):
        ''' Performs a visualization step for the data.

        Args:
            data (dict): data dictionary
        '''
        device = self.device

        batch_size = data['points'].size(0)
        inputs = data.get('inputs', torch.empty(batch_size, 0)).to(device)

        shape = (32, 32, 32)
        p = make_3d_grid([-0.5] * 3, [0.5] * 3, shape).to(device)
        p = p.expand(batch_size, *p.size())

        kwargs = {}
        with torch.no_grad():
            p_out0, p_out1, p_out2, p_out3, p_out4 = self.model(p, inputs, sample=self.eval_sample, **kwargs)

        occ_hat = p_out4.probs.view(batch_size, *shape)
        voxels_out = (occ_hat >= self.threshold).cpu().numpy()

        for i in trange(batch_size):
            input_img_path = os.path.join(self.vis_dir, '%03d_in.png' % i)
            vis.visualize_data(
                inputs[i].cpu(), self.input_type, input_img_path)
            vis.visualize_voxels(
                voxels_out[i], os.path.join(self.vis_dir, '%03d.png' % i))

    def compute_loss(self, data):
        ''' Computes the loss.

        Args:
            data (dict): data dictionary
        '''
        device = self.device
        p = data.get('points').to(device)
        occ = data.get('points.occ').to(device)
        inputs = data.get('inputs', torch.empty(p.size(0), 0)).to(device)

        kwargs = {}

        c0, c1, c2, cdog = self.model.encode_inputs(inputs)
        q_z = self.model.infer_z(p, occ, c0, **kwargs)
        z = q_z.rsample()

        # KL-divergence
        kl = dist.kl_divergence(q_z, self.model.p0_z).sum(dim=-1)
        loss = kl.mean()

        # General points
        logits0, logits1, logits2, logits3, logits4 = self.model.decode(p, z, c0, c1, c2, cdog, **kwargs)
        loss_i0 = F.binary_cross_entropy_with_logits(
            logits0.logits, occ, reduction='none')

        loss_i1 = F.binary_cross_entropy_with_logits(
            logits1.logits, occ, reduction='none')

        loss_i2 = F.binary_cross_entropy_with_logits(
            logits2.logits, occ, reduction='none')

        loss_i3 = F.binary_cross_entropy_with_logits(
            logits3.logits, occ, reduction='none')

        loss_i4 = F.binary_cross_entropy_with_logits(
            logits4.logits, occ, reduction='none')

        loss = loss + loss_i0.sum(-1).mean() + loss_i1.sum(-1).mean() + loss_i2.sum(-1).mean() + loss_i3.sum(
            -1).mean() + loss_i4.sum(-1).mean()

        return loss, loss_i0.sum(-1).mean(), loss_i1.sum(-1).mean(), loss_i2.sum(-1).mean(), loss_i3.sum(
            -1).mean(), loss_i4.sum(-1).mean()
