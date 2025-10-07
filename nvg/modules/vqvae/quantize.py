import math
import os
from collections import defaultdict
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch._dynamo
from einops import rearrange


compile_mode = os.getenv("USE_TORCH_COMPILE", "1") == "1"
print("Quantizer Compile mode:", compile_mode)

def maybe_compile(fn):
    if compile_mode:
        return torch.compile(fn)
    else:
        return fn

def maybe_dynamo_optimize(fn):
    if compile_mode:
        return torch._dynamo.optimize("inductor")(fn)
    else:
        return fn

def pair_points(points_batch, iter_num, lab2id, reduce_factor):
    batch_size = points_batch.size(0)
    num_points = points_batch.size(1)
    device = points_batch.device

    inf_value = 1e8

    distance_matrix = torch.cdist(points_batch, points_batch, p=2)
    distance_matrix += torch.eye(num_points, device=device).unsqueeze(0).expand(batch_size, -1, -1) * inf_value

    cluster_num = reduce_factor ** iter_num
    new_lab2id = torch.empty((batch_size, num_points // reduce_factor, cluster_num), dtype=torch.long, device=device)
    new_id2lab = torch.empty((batch_size, num_points * cluster_num // reduce_factor), dtype=torch.long, device=device)

    for match_id in range(num_points // reduce_factor):
        row_indices = torch.arange(batch_size, device=device)

        _, min_idx = torch.min(distance_matrix.view(batch_size, -1), dim=1)
        i = min_idx // num_points
        j = min_idx % num_points

        ids_i = lab2id[row_indices, i]
        ids_j = lab2id[row_indices, j]

        new_lables = torch.ones(batch_size, cluster_num // reduce_factor, dtype=torch.long, device=device) * match_id
        new_id2lab.scatter_(1, ids_i, new_lables)
        new_id2lab.scatter_(1, ids_j, new_lables)


        new_lab2id[:, match_id, :(cluster_num // reduce_factor)] = ids_i
        new_lab2id[:, match_id, (cluster_num // reduce_factor):] = ids_j


        distance_matrix[row_indices, i, :] = inf_value
        distance_matrix[row_indices, :, i] = inf_value
        distance_matrix[row_indices, j, :] = inf_value
        distance_matrix[row_indices, :, j] = inf_value

    return new_lab2id, new_id2lab

def pr2vis(lab2id, colors, channels):
    # pair results -> colors
    bs, label_num, cluster_num = lab2id.shape
    device = lab2id.device

    new_colors = torch.zeros(bs, label_num * cluster_num, channels, device=device).to(colors.dtype)

    for i in range(label_num):
        ids = lab2id[:, i, :].unsqueeze(-1).expand(-1, -1, channels)
        target_color = torch.gather(colors, 1, ids)
        target_color = torch.mean(target_color, 1)
        new_colors.scatter_(1, ids, target_color.unsqueeze(1).repeat(1, cluster_num, 1))

    return new_colors

@maybe_dynamo_optimize
def clustering_points(points):
    bs, nums, channels = points.shape
    device = points.device

    reduce_factor = 2

    iter_nums = int(math.log(nums, reduce_factor))
    all_labels = torch.arange(nums, device=device).unsqueeze(0).unsqueeze(0).repeat(bs, iter_nums+1, 1)

    cur_points = points.clone()
    id2lab = torch.arange(nums, device=device).unsqueeze(0).repeat(bs, 1)
    lab2id = torch.arange(nums, device=device).unsqueeze(0).repeat(bs, 1).unsqueeze(-1)  # bs, label_num, cluster_num

    for i in range(1, iter_nums + 2):
        all_labels[:, iter_nums+1-i, :] = id2lab

        if i == iter_nums + 1:
            break

        ids = lab2id[:, :, 0].unsqueeze(-1).expand(-1, -1, channels)
        cur_points = torch.gather(cur_points, 1, ids)
        lab2id, id2lab = pair_points(cur_points, i, lab2id, reduce_factor)
        cur_points = pr2vis(lab2id, points, channels)
    all_labels = reorganize(all_labels)
    return all_labels

def reorganize(cluster_tensor):
    reorganized = torch.zeros_like(cluster_tensor)
    B, L, N = cluster_tensor.shape
    for level in range(1, L):
        parent2token = reorganized[:, level - 1]
        child2token = cluster_tensor[:, level]

        children = torch.empty(B, 2**level, device=cluster_tensor.device, dtype=cluster_tensor.dtype)
        children.scatter_(1, child2token, parent2token)

        current_parent2child = torch.sort(children, dim=1, stable=True)[1]

        # Simplified inner loop
        new_indices = torch.empty_like(current_parent2child)
        new_indices.scatter_(1, current_parent2child, torch.arange(current_parent2child.size(1), device=cluster_tensor.device).unsqueeze(0).repeat(B, 1))
        reorganized[:, level] = torch.gather(new_indices, 1, child2token)

    return reorganized

class Phi(nn.Conv2d):
    def __init__(self, embed_dim, quant_resi):
        ks = 3
        super().__init__(in_channels=embed_dim, out_channels=embed_dim, kernel_size=ks, stride=1, padding=ks//2)
        self.resi_ratio = abs(quant_resi)

    @maybe_compile
    def forward(self, h_BChw):
        return h_BChw + super().forward(h_BChw).mul_(self.resi_ratio)

class PhiShared(nn.Module):
    def __init__(self, qresi: Phi):
        super().__init__()
        self.qresi: Phi = qresi

    def __getitem__(self, _) -> Phi:
        return self.qresi

class PhiPartiallyShared(nn.Module):
    def __init__(self, qresi_ls: nn.ModuleList):
        super().__init__()
        self.qresi_ls = qresi_ls
        K = len(qresi_ls)
        self.ticks = np.linspace(1/3/K, 1-1/3/K, K) if K == 4 else np.linspace(1/2/K, 1-1/2/K, K)

    def __getitem__(self, at_from_0_to_1: float) -> Phi:
        return self.qresi_ls[np.argmin(np.abs(self.ticks - at_from_0_to_1)).item()]

    def extra_repr(self) -> str:
        return f'ticks={self.ticks}'

class PhiNonShared(nn.ModuleList):
    def __init__(self, qresi: List):
        super().__init__(qresi)
        # self.qresi = qresi
        K = len(qresi)
        self.ticks = np.linspace(1/3/K, 1-1/3/K, K) if K == 4 else np.linspace(1/2/K, 1-1/2/K, K)

    def __getitem__(self, at_from_0_to_1: float) -> Phi:
        return super().__getitem__(np.argmin(np.abs(self.ticks - at_from_0_to_1)).item())

    def extra_repr(self) -> str:
        return f'ticks={self.ticks}'

class VQ2(nn.Module):
    def __init__(self, n_e, e_dim, beta, remap=None, unknown_index="random",
                 sane_index_shape=False,
                 quant_resi=0.5, share_quant_resi=0, default_qresi_counts=0,
                 v_patch_nums=None, smooth_end_epoch=-1,
                 ):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta

        self.v_patch_nums = v_patch_nums
        self.smooth_end_epoch = smooth_end_epoch

        # conv for addressing information loss
        self.quant_resi_ratio = quant_resi
        if share_quant_resi == 0:   # non-shared: \phi_{1 to K} for K scales
            self.quant_resi = PhiNonShared([(Phi(self.e_dim, quant_resi) if abs(quant_resi) > 1e-6 else nn.Identity()) for _ in range(default_qresi_counts or len(self.v_patch_nums))])
        elif share_quant_resi == 1: # fully shared: only a single \phi for K scales
            self.quant_resi = PhiShared(Phi(self.e_dim, quant_resi) if abs(quant_resi) > 1e-6 else nn.Identity())
        else:                       # partially shared: \phi_{1 to share_quant_resi} for K scales
            self.quant_resi = PhiPartiallyShared(nn.ModuleList([(Phi(self.e_dim , quant_resi) if abs(quant_resi) > 1e-6 else nn.Identity()) for _ in range(share_quant_resi)]))

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

        self.remap = remap
        if self.remap is not None:
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            self.re_embed = self.used.shape[0]
            self.unknown_index = unknown_index # "random" or "extra" or integer
            if self.unknown_index == "extra":
                self.unknown_index = self.re_embed
                self.re_embed = self.re_embed+1
            print(f"Remapping {self.n_e} indices to {self.re_embed} indices. "
                  f"Using {self.unknown_index} for unknown indices.")
        else:
            self.re_embed = n_e

        self.sane_index_shape = sane_index_shape

    def remap_to_used(self, inds):
        ishape = inds.shape
        assert len(ishape)>1
        inds = inds.reshape(ishape[0],-1)
        used = self.used.to(inds)
        match = (inds[:,:,None]==used[None,None,...]).long()
        new = match.argmax(-1)
        unknown = match.sum(2)<1
        if self.unknown_index == "random":
            new[unknown]=torch.randint(0,self.re_embed,size=new[unknown].shape).to(device=new.device)
        else:
            new[unknown] = self.unknown_index
        return new.reshape(ishape)

    def unmap_to_all(self, inds):
        ishape = inds.shape
        assert len(ishape)>1
        inds = inds.reshape(ishape[0],-1)
        used = self.used.to(inds)
        if self.re_embed > self.used.shape[0]: # extra token
            inds[inds>=self.used.shape[0]] = 0 # simply set to zero
        back=torch.gather(used[None,:][inds.shape[0]*[0],:], 1, inds)
        return back.reshape(ishape)

    def forward(self, z, temp=None, rescale_logits=False, return_logits=False, epoch=None):
        assert temp is None or temp==1.0, "Only for interface compatible with Gumbel"
        assert rescale_logits==False, "Only for interface compatible with Gumbel"
        assert return_logits==False, "Only for interface compatible with Gumbel"
        f_BChw = z
        dtype = f_BChw.dtype
        if dtype != torch.float32:
            f_BChw = f_BChw.float()
        B, C, H, W = f_BChw.shape
        f_no_grad = f_BChw.detach()

        f_rest = f_no_grad.clone()

        # modified from here
        f_hat = torch.zeros_like(f_rest)
        # set gts for each scale
        gt_labels = clustering_points(f_rest.permute(0, 2, 3, 1).reshape(B, H*W, C))

        embedding = self.embedding.weight

        mean_vq_loss: torch.Tensor = 0.0
        min_encoding_indices = []
        SN = len(self.v_patch_nums)

        vq_loss_dict = defaultdict(float)
        for si, pn in enumerate(self.v_patch_nums): # from small to large
            # find the nearest embedding
            label_map = gt_labels[:, si, :] # [B, L]
            # Expand labels to match the input tensor shape (B, L, C)
            labels_one_hot = torch.nn.functional.one_hot(label_map, num_classes=pn).float()  # (B, L, N)
            # Transpose labels_one_hot to (B, N, L) so that we can broadcast multiply
            labels_one_hot = labels_one_hot.permute(0, 2, 1)  # (B, N, L)
            # Perform batch matrix multiplication: (B, N, L) x (B, L, C) -> (B, N, C)
            # This effectively groups by the labels and sums the values
            class_sums = torch.bmm(labels_one_hot, f_rest.permute(0, 2, 3, 1).reshape(B, H*W, C))  # (B, N, C)
            # Count the occurrences of each class to normalize (get the averages)
            class_counts = labels_one_hot.sum(dim=-1, keepdim=True).clamp(min=1)  # (B, N, 1) to avoid division by zero
            # Calculate the average by dividing summed values by the count
            rest_NC = class_sums / class_counts  # (B, N, C)
            rest_NC = rest_NC.reshape(-1, C)

            d_no_grad = torch.sum(rest_NC.square(), dim=1, keepdim=True) + torch.sum(embedding.square(), dim=1, keepdim=False)
            d_no_grad.addmm_(rest_NC, embedding.T, alpha=-2, beta=1)  # (B*h*w, vocab_size)

            if epoch is not None and self.smooth_end_epoch != -1 and epoch < self.smooth_end_epoch:
                d_no_grad = d_no_grad.reshape(B, -1, self.n_e)
                d_no_grad = d_no_grad.max(dim=-1, keepdim=True)[0] - d_no_grad
                d_no_grad = d_no_grad.reshape(-1, self.n_e)
                logit = F.softmax(d_no_grad, dim=-1)
                idx_N = torch.argmax(logit, dim=-1)

                min_encoding_indices.append(idx_N)
                hard_one_hot = F.one_hot(idx_N, num_classes=self.n_e).to(logit.dtype).to(logit.device)
                one_hot = hard_one_hot - logit.detach() + logit
                h_BNC = torch.einsum('nv,vc->nc', one_hot, embedding)
            else:
                idx_N = torch.argmin(d_no_grad, dim=1)
                min_encoding_indices.append(idx_N)
                h_BNC = embedding[idx_N]
            # calc loss
            h_BNC = h_BNC.reshape(B, -1, C)
            h_BChw = torch.gather(h_BNC, 1, label_map.unsqueeze(-1).expand(-1, -1, C)).reshape(B, H, W, C).permute(0, 3, 1, 2) # [B, L]
            h_BChw = self.quant_resi[si/(SN-1)](h_BChw)

            f_hat = f_hat + h_BChw
            f_rest -= h_BChw


            mean_vq_loss_i = F.mse_loss(f_hat.data, f_BChw).mul_(self.beta) + F.mse_loss(f_hat, f_no_grad)
            vq_loss_dict[f'vq_loss_{si}'] = mean_vq_loss_i.item()
            mean_vq_loss += mean_vq_loss_i

        mean_vq_loss *= 1. / SN
        f_hat = (f_hat.data - f_no_grad).add_(f_BChw)

        return f_hat, mean_vq_loss, (None, None, min_encoding_indices), vq_loss_dict


    def f_to_nvg_input_output(self, f, predict_final=False):
        B, C, H, W = f.shape
        L = H * W
        f_rest = f.detach().clone().to(torch.float32)
        f_hat = torch.zeros_like(f_rest)
        SN = len(self.v_patch_nums)

        nvg_input = torch.full((B, SN, L, C+2), -1, device=f.device, dtype=torch.float32)
        nvg_input[:, 0, :,:C] = 0 # empty canvas

        gt_labels = clustering_points(f_rest.permute(0, 2, 3, 1).reshape(B, L, C))

        embedding = self.embedding.weight

        for si, pn in enumerate(self.v_patch_nums): # from small to large
            label_map = gt_labels[:, si, :]
            labels_one_hot = torch.nn.functional.one_hot(label_map, num_classes=pn).float()
            labels_one_hot = labels_one_hot.permute(0, 2, 1)
            class_sums = torch.bmm(labels_one_hot, f_rest.permute(0, 2, 3, 1).reshape(B, L, C))
            class_counts = labels_one_hot.sum(dim=-1, keepdim=True).clamp(min=1)
            rest_NC = class_sums / class_counts
            rest_NC = rest_NC.reshape(-1, C)
            z_NC = rest_NC

            d_no_grad = torch.sum(z_NC.square(), dim=1, keepdim=True) + torch.sum(embedding.square(), dim=1, keepdim=False)
            d_no_grad.addmm_(z_NC, embedding.T, alpha=-2, beta=1)

            idx_N = torch.argmin(d_no_grad, dim=1)

            h_BNC = embedding[idx_N]
            h_BNC = h_BNC.reshape(B, -1, C)
            h_BL = torch.gather(h_BNC, 1, label_map.unsqueeze(-1).expand(-1, -1, C))
            idx_BL = torch.gather(idx_N.reshape(B, -1), 1, label_map)

            h_BChw = h_BL.reshape(B, H, W, C).permute(0, 3, 1, 2)
            h_BChw = self.quant_resi[si/(SN-1)](h_BChw)
            f_hat.add_(h_BChw)

            nvg_input[:, si, :, C] = idx_BL
            nvg_input[:, si, :, C+1] = label_map
            if si < SN-1:
                nvg_input[:, si+1, :, :C] = rearrange(f_hat, 'b c h w -> b (h w) c')

            f_rest.sub_(h_BChw)

        if predict_final:
            return nvg_input, rearrange(f_hat, 'b c h w -> b (h w) c')
        return nvg_input

    def nvg_output_to_fhat(self, nvg_output, full_list=False):
        nvg_output = nvg_output.long()
        B = nvg_output.shape[0]
        L = self.v_patch_nums[-1]
        H = W = int(math.sqrt(L))
        SN = len(self.v_patch_nums)
        f_hat = torch.zeros(B, self.e_dim, H, W, device=nvg_output.device, dtype=torch.float32)
        embedding = self.embedding.weight
        if full_list:
            f_hats = []
        for si in range(SN):
            h_BLC = embedding[nvg_output[:, si, :]].float()
            h_BChw = h_BLC.reshape(B, H, W, self.e_dim).permute(0, 3, 1, 2)
            h_BChw = self.quant_resi[si/(SN-1)](h_BChw)
            f_hat.add_(h_BChw)
            if full_list:
                f_hats.append(f_hat.clone())
        if full_list:
            return f_hats
        else:
            return f_hat

    def nvg_next_input(self, inp, out, stage):
        B = out.shape[0]
        L = self.v_patch_nums[-1]
        H = W = int(math.sqrt(L))
        nvg_output = out.long()
        embedding = self.embedding.weight
        f_hat = inp
        f_hat = rearrange(f_hat, 'B (H W) C -> B C H W', H=H)
        h_BLC = embedding[nvg_output]
        h_BChw = h_BLC.reshape(B, H, W, self.e_dim).permute(0, 3, 1, 2)
        h_BChw = self.quant_resi[stage/(len(self.v_patch_nums)-1)](h_BChw)
        f_hat.add_(h_BChw)
        next_input = f_hat.permute(0, 2, 3, 1).reshape(B, H*W, self.e_dim)
        return next_input