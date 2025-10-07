import os
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from collections import OrderedDict

from main import instantiate_from_config
from copy import deepcopy
from nvg.modules.generatormodules.repa import load_encoders, preprocess_raw_image


compile_mode = os.getenv("USE_TORCH_COMPILE", "1") == "1"
print("REPA Compile mode:", compile_mode)

def maybe_compile(fn):
    if compile_mode:
        return torch.compile(fn)
    else:
        return fn

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return True

class NVGenerator(pl.LightningModule):
    def __init__(self,
                 nvgformer_config,
                 first_stage_config,
                 cond_stage_config=None,
                 downsample_cond_size=-1,
                 cond_stage_trainable=False,
                 cfg=False,
                 cfg_rate=0,
                 use_ema=False,
                 loss_weight_strategy='uniform',
                 use_repa=False,
                 repa_depth=8,
                 final_loss_scale=1.0,
                 next_unique_loss_scale=1.0,
                 repa_loss_scale=1.0,
                 structure_loss_scale=1.0,
                 ckpt_path=None,
                 load_ema=False,
                 ignore_keys=[],
                 ):
        super(NVGenerator, self).__init__()
        self.use_repa = use_repa
        self.final_loss_scale = final_loss_scale
        self.next_unique_loss_scale = next_unique_loss_scale
        self.structure_loss_scale = structure_loss_scale
        self.repa_loss_scale = repa_loss_scale
        self.init_first_stage_from_ckpt(first_stage_config)
        self.cond_stage_config = cond_stage_config
        if self.cond_stage_config is not None:
            self.init_cond_stage_from_ckpt(cond_stage_config)
        self.nvgformer = instantiate_from_config(config=nvgformer_config)
        self.token_sequence = self.nvgformer.num_tokens
        self.n_stage = len(self.token_sequence)
        self.downsample_cond_size = downsample_cond_size
        self.cond_stage_trainable = cond_stage_trainable
        self.cfg = cfg
        self.cfg_rate = cfg_rate
        if self.use_repa:
            self.init_repa(repa_depth=repa_depth)
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys, load_ema=load_ema)
        self.use_ema = use_ema
        if self.use_ema:
            self.init_ema()
        self.init_loss_weight(loss_weight_strategy)
        self.nvgformer.nvg_next_input = self.first_stage_model.nvg_next_input

    def init_from_ckpt(self, path, ignore_keys=list(), load_ema=False):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        if load_ema:
            print("Loading EMA model from checkpoint.")
            keys = list(sd.keys())
            for k in keys:
                if k.startswith("ema_model"):
                    sd[k.replace("ema_model.", "")] = sd[k]
                    del sd[k]
        else:
            print("Loading model from checkpoint without EMA.")
            keys = list(sd.keys())
            for k in keys:
                if k.startswith("ema_model"):
                    del sd[k]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    del sd[k]
        sd = {k.split('nvgformer.')[1]: v for k, v in sd.items()}
        missing, unexpected = self.nvgformer.load_state_dict(sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected keys: {unexpected}")

    def init_loss_weight(self, strategy):
        if strategy == 'uniform':
            self.loss_weight = torch.tensor([1.0] * self.n_stage, device=self.device)
            self.loss_weight_structure = self.loss_weight[:-2].clone() * self.n_stage / (self.n_stage - 2)
        elif strategy == 'arccos':
            self.loss_weight = torch.arange(self.n_stage, device=self.device)
            self.loss_weight = torch.arccos(self.loss_weight / self.n_stage)
            self.loss_weight = self.n_stage * self.loss_weight / self.loss_weight.sum()
            self.loss_weight_structure = self.loss_weight[:-2].clone() * self.n_stage / (self.n_stage - 2)
        else:
            raise ValueError(f"Unknown loss weight strategy: {strategy}")
        print(f"loss weight: {self.loss_weight}")
        print(f"loss weight structure: {self.loss_weight_structure}")

    def init_repa(self, repa_depth=6):
        encoder = load_encoders()
        self.repa_encoder = encoder
        for param in self.repa_encoder.parameters():
            param.requires_grad = False
        self.nvgformer.set_repa_alignment_layer(repa_depth=repa_depth, repa_dims=self.repa_encoder.embed_dim) # TODO: ablate

    def init_ema(self):
        # only copy nvgformer and cond_stage_model into a single ema model
        self.ema_update_step = 0
        self.ema_model = nn.Module()
        self.ema_model.nvgformer = deepcopy(self.nvgformer).to(self.device)
        if self.cond_stage_config is not None:
            self.ema_model.cond_stage_model = deepcopy(self.cond_stage_model).to(self.device)
        self.ema_model = self.ema_model.eval()
        self.ema_model.train = disabled_train
        for param in self.ema_model.parameters():
            param.requires_grad = False

    def init_first_stage_from_ckpt(self, config):
        model = instantiate_from_config(config)
        model = model.eval()
        model.train = disabled_train
        self.first_stage_model = model
        for param in self.first_stage_model.parameters():
            param.requires_grad = False

    def init_cond_stage_from_ckpt(self, config):
        if config == "__is_first_stage__":
            print("Using first stage also as cond stage.")
            self.cond_stage_model = self.first_stage_model
        else:
            model = instantiate_from_config(config)
            self.cond_stage_model = model

    @torch.no_grad()
    def update_ema(self, decay=0.9999):
        """
        Step the EMA model towards the current model.
        """
        ema_params = OrderedDict(self.ema_model.named_parameters())
        model_params = OrderedDict(self.named_parameters())
        for name in ema_params.keys():
            ema_params[name].mul_(decay).add_(model_params[name].detach(), alpha=1 - decay)
        self.ema_update_step = self.global_step

    def forward(self, batch):
        img, final_target, repa_target = self.encode_to_z(batch['image']) # B, L, 2
        txt = batch['class_label']
        if self.cfg and self.train:
            # used for classifier free guidance
            rand_cfg = torch.rand(txt.size(), device=txt.device)
            txt = torch.where(rand_cfg < self.cfg_rate, 1000, txt)
        txt = self.encode_to_c(txt)
        img_inp, img_out, structure_id = img[:, :, :, :-2], img[:, :, :, -2], img[:, :, :, -1]
        final_hat, compressed_logits, structure_logits, repa_features, structure_target = self.nvgformer(img=img_inp, txt=txt, structure=structure_id)
        return final_hat, final_target, compressed_logits, img_out, structure_logits, structure_id, repa_features, repa_target, structure_target

    def encode_to_c(self, c):
        if self.downsample_cond_size > -1:
            c = F.interpolate(c, size=(self.downsample_cond_size, self.downsample_cond_size))
        c = c.to(self.device)[:, None]
        if self.cond_stage_config is not None:
            c = self.cond_stage_model(c)
        return c

    @maybe_compile
    def get_repa_features(self, x):
        x = preprocess_raw_image(x)
        repa_features = self.repa_encoder.forward_features(x)['x_norm_patchtokens']
        return repa_features

    @torch.no_grad()
    def encode_to_z(self, x):
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float().to(self.device)
        z, fhat = self.first_stage_model.img_to_nvg_input_output(x, predict_final=True) # B, S, L, C+2
        if self.use_repa:
            repa_target = self.get_repa_features(x)
        else:
            repa_target = None
        return z, fhat, repa_target

    @torch.no_grad()
    def log_images(self, batch, max_images, **kwargs):
        if not self.pred_content:
            return {"gen_images": torch.zeros((max_images, 3, 256, 256))}
        batch_size = max_images
        device = batch['image'].device
        class_label = torch.randint(0, 1000, (batch_size, ), device=device)
        structure_noise = torch.randn((batch_size, 256, 8), device=device)

        images_and_structures = self.generate_images(class_label=class_label,
                                    structure_noise=structure_noise,
                                    content_use_cfg=False,
                                    content_cfg_scale=[1.0]*self.n_stage,
                                    structure_use_cfg=False,
                                    structure_cfg_scale=[1.0]*self.n_stage,
                                    structure_sampling_step=50,
                                    top_k=0,
                                    temperature=1.0,
                                    top_p=1.0,
                                    full_list=False,
                                    return_structure=True,
                                    )

        log = {"gen_images": images_and_structures[1]}
        return log

    def combine_image_and_structure(self, image, structure, B):
        structure = [torch.zeros_like(structure[-1])] + structure
        structure = torch.cat(structure, dim=0).float() # (B I) L
        structure = rearrange(structure.detach().cpu(), "B (H W)-> B 1 H W", H=int(self.token_sequence[-1]**0.5))
        structure = torch.nn.functional.interpolate(structure, size=(256, 256), mode="nearest").repeat(1, 3, 1, 1)
        structure = rearrange(structure, "(I B) C H W -> B I C H W ", B=B)
        image_and_structure = torch.cat([structure, image], dim=-2)
        image_and_structure = rearrange(image_and_structure, "B I C H W -> B C H (I W) ", B=B)
        return image_and_structure

    @torch.no_grad()
    def generate_images(self, class_label, structure_noise, content_use_cfg=False, content_cfg_scale=1.0, structure_use_cfg=False, structure_cfg_scale=1.0, structure_sampling_step=50, top_k=0, temperature=1.0, top_p=1.0, full_list=False, return_structure=False, use_gumbel_topk=False):
        txt = self.encode_to_c(class_label)
        if full_list:
            image_list = []
            if return_structure:
                structure_list = []
        B = class_label.size(0)
        image = torch.zeros((B, self.token_sequence[-1], 32), device=class_label.device)
        structure = torch.zeros((B, self.token_sequence[-1]), device=class_label.device)
        if content_use_cfg:
            content_cfg_scale = torch.tensor(content_cfg_scale, device=class_label.device, dtype=torch.bfloat16)
        if structure_use_cfg:
            structure_cfg_scale = torch.tensor(structure_cfg_scale, device=class_label.device, dtype=torch.bfloat16)

        for i in range(self.n_stage):
            with torch.autocast('cuda', enabled=True, dtype=torch.bfloat16):
                _, _, image, structure = self.nvgformer.generate(img=image,
                                                                  txt=txt,
                                                                  stage=i,
                                                                  structure=structure,
                                                                  structure_epsilon=structure_noise,
                                                                  top_k=top_k[i],
                                                                  temperature=temperature,
                                                                  top_p=top_p[i],
                                                                  content_use_cfg=content_use_cfg,
                                                                  content_cfg_scale=content_cfg_scale[i],
                                                                  structure_use_cfg=structure_use_cfg,
                                                                  structure_cfg_scale=structure_cfg_scale[i],
                                                                  structure_sampling_step=structure_sampling_step,
                                                                  use_gumbel_topk=use_gumbel_topk,
                                                                  )
            if full_list:
                image_list.append(image.clone())
                if return_structure and i < self.n_stage - 1:
                    structure_list.append((structure.clone()/self.token_sequence[i+1])*2-1)
            elif return_structure and i==0:
                returned_structure = (structure.clone()/self.token_sequence[i+1]+1)*2-1
        if not full_list:
            image = rearrange(image, "B (H W) C -> B C H W", H=int(self.token_sequence[-1]**0.5))
            image = self.first_stage_model.fhat_to_img(image).detach().cpu() # (B I) C H W
            if return_structure:
                structure = rearrange(returned_structure.detach().cpu(), "B (H W)-> B 1 H W", H=int(self.token_sequence[-1]**0.5)).float()
                structure = torch.nn.functional.interpolate(structure, size=(256, 256), mode="nearest").repeat(1, 3, 1, 1)
                image_and_structure = torch.cat([structure, image], dim=-1)
        else:
            image_list = torch.cat(image_list, dim=0)
            image_list = rearrange(image_list, "B (H W) C -> B C H W", H=int(self.token_sequence[-1]**0.5))
            image_list = self.first_stage_model.fhat_to_img(image_list).detach().cpu()
            image_list = rearrange(image_list, "(I B) C H W -> B I C H W ", B=B)
            image = image_list[:, -1]
            if return_structure:
                image_and_structure = self.combine_image_and_structure(image_list, structure_list, B)
        if not full_list:
            if return_structure:
                return image, image_and_structure
            else:
                return image
        else:
            if return_structure:
                return image, image_and_structure
            else:
                return image, image_list

    def stablize_logits(self,logits):
        logits_max, _ = torch.max(logits, dim=-1, keepdim=True)
        logits = logits - logits_max.detach()
        return logits

    def get_compressed_target(self, target, structure, stage):
        compressed_target = torch.empty(target.shape[0], self.token_sequence[stage], dtype=torch.long, device=target.device)
        compressed_target.scatter_(1, structure, target)
        return compressed_target

    def repa_loss(self, repa_features, repa_target):
        repa_features = torch.nn.functional.normalize(repa_features, dim=-1)
        repa_target = torch.nn.functional.normalize(repa_target, dim=-1)
        repa_loss = 1 - torch.einsum("l c, l c -> l", repa_features, repa_target).mean()
        return repa_loss

    def shared_full_step(self, batch, batch_idx, tag=None):
        final_hat, final_target, compressed_logits, target, structure_logits, structure, repa_features, repa_target, structure_target = self(batch)
        B = target.size(0)
        target = rearrange(target, "B S L -> S (B L)", B=B).long()
        structure = rearrange(structure, "B S L -> S B L", B=B).long()
        final_hat = rearrange(final_hat, "(B S) L D -> S B L D", B=B)

        structure_logits = rearrange(structure_logits, "(B S) L C -> S B L C", B=B)
        structure_target = rearrange(structure_target, "B S L C-> S B L C")

        if self.use_repa:
            repa_features = rearrange(repa_features, "(B S) L C -> S (B L) C", B=B)
            repa_target = rearrange(repa_target, "B L C -> (B L) C")
        loss = 0
        log_dict = {}
        for i in range(self.n_stage):

            final_loss_i = F.mse_loss(final_hat[i], final_target)
            loss += final_loss_i * self.loss_weight[i] * self.final_loss_scale
            log_dict[f"final_loss_{self.token_sequence[i]}"] = final_loss_i.item()

            target_i = rearrange(target[i], "(B L) -> B L", B=B)
            compressed_target_i = self.get_compressed_target(target_i, structure[i], i)
            compressed_target_i = rearrange(compressed_target_i, "B N -> (B N)")
            compressed_logits_i = rearrange(compressed_logits[i], "B N C -> (B N) C")
            compressed_logits_i = self.stablize_logits(compressed_logits_i)
            compressed_loss_i = F.cross_entropy(compressed_logits_i, compressed_target_i)
            loss += compressed_loss_i * self.loss_weight[i] * self.next_unique_loss_scale
            log_dict[f"compressed_loss_{self.token_sequence[i]}"] = compressed_loss_i.item()
            log_dict[f"compressed_acc_{self.token_sequence[i]}"] = (compressed_logits_i.argmax(-1) == compressed_target_i).float().mean().item()

            if i < self.n_stage - 2:
                structure_loss_i = F.mse_loss(structure_logits[i], structure_target[i])
                loss += structure_loss_i * self.loss_weight_structure[i] * self.structure_loss_scale
                log_dict[f"structure_loss_{self.token_sequence[i+1]}"] = structure_loss_i.item()

            if self.use_repa:
                repa_loss_i = self.repa_loss(repa_features[i], repa_target)
                loss += repa_loss_i * self.loss_weight[i] * self.repa_loss_scale
                log_dict[f"repa_loss_{self.token_sequence[i]}"] = repa_loss_i.item()

        loss = loss / self.n_stage
        log_dict["loss"] = loss.item()
        log_dict["compressed_acc"] = np.mean([log_dict[f"compressed_acc_{self.token_sequence[i]}"] for i in range(self.n_stage)])
        for k, v in log_dict.items():
            self.log(f"{tag}/{k}", v, prog_bar=False, logger=True, on_step=True if tag=='train' else False, on_epoch=True if tag=='val' else False, sync_dist=False, batch_size=B)
        return loss

    def training_step(self, batch, batch_idx):
        if self.use_ema and self.global_step > self.ema_update_step:
            self.update_ema()
        return self.shared_full_step(batch, batch_idx, tag="train")

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        return self.shared_full_step(batch, batch_idx, tag="val")

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(filter(lambda p: p.requires_grad, self.nvgformer.parameters()))
        if self.cond_stage_trainable:
            print(f"{self.__class__.__name__}: Also optimizing conditioner params!")
            params = params + list(self.cond_stage_model.parameters())

        decay_params = [p for p in params if p.dim() >= 2]
        nodecay_params = [p for p in params if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": 5e-2},
            {"params": nodecay_params, "weight_decay": 0.0}
        ]
        opt = torch.optim.AdamW(optim_groups, lr=1, betas=(0.9, 0.95))
        total_steps = self.trainer.estimated_stepping_batches
        steps_per_epoch = self.trainer.estimated_stepping_batches // self.trainer.max_epochs
        warmup_steps = 1000
        hold_steps = 200 * steps_per_epoch - warmup_steps if self.trainer.max_epochs > 300 else int(0.8 * total_steps)
        start_lr = 0.005 * lr
        base_lr = lr
        final_lr = 0.1 * lr if self.trainer.max_epochs < 300 else 0.01 * lr
        print(f"Using three-phase schedule with warmup {warmup_steps}, hold {hold_steps}, total {total_steps}, start_lr {start_lr}, base_lr {base_lr}, final_lr {final_lr}")

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            opt,
            lr_lambda=three_phase_schedule_fn(
                warmup_steps=warmup_steps,
                hold_steps=hold_steps,
                total_steps=total_steps,
                start_lr=start_lr,
                base_lr=base_lr,
                final_lr=final_lr
            )
        )

        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",  # important for fine-grained control
                "frequency": 1,
            },
        }

def three_phase_schedule_fn(
    warmup_steps: int,
    hold_steps: int,
    total_steps: int,
    start_lr: float,
    base_lr: float,
    final_lr: float,
):
    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            # Linear warmup: start_lr -> base_lr
            return start_lr + (base_lr - start_lr) * (current_step / warmup_steps)
        elif current_step < (warmup_steps + hold_steps):
            # Constant hold
            return base_lr
        elif current_step < total_steps:
            # Linear decay: base_lr -> final_lr
            decay_steps = total_steps - (warmup_steps + hold_steps)
            step_into_decay = current_step - (warmup_steps + hold_steps)
            return base_lr - (base_lr - final_lr) * (step_into_decay / decay_steps)
        else:
            return final_lr  # After training is over

    return lr_lambda
