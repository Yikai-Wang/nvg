import os
from copy import deepcopy
from collections import OrderedDict

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.utilities import rank_zero_only

from main import instantiate_from_config
from nvg.modules.vqvae.ae import Encoder, Decoder
from nvg.modules.vqvae.quantize import VQ2


compile_mode = os.getenv("USE_TORCH_COMPILE", "1") == "1"
print("AE Compile mode:", compile_mode)

def maybe_compile(fn):
    if compile_mode:
        return torch.compile(fn)
    else:
        return fn

class VQAE(pl.LightningModule):
    def __init__(self,
                 n_embed,
                 embed_dim,
                 lossconfig=None,
                 v_patch_nums=(1, 2, 4, 8, 16),
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 remap=None,
                 sane_index_shape=False,
                 cnn_type='2d',
                 conv_in_out_2d='no',
                 res_conv_2d='no',
                 cnn_attention='no',
                 cnn_norm_axis='spatial',
                 conv_inner_2d='no',
                 base_ch=None,
                 encoder_ch_mult=None,
                 decoder_ch_mult=None,
                 num_res_blocks=None,
                 downsampled_size=None,
                 temporal_patch_size=None,
                 use_checkpoint=False,
                 use_vae=False,
                 use_freq_dec=False,
                 use_pxsf=False,
                 gradient_accumulation_steps=4,
                 gradient_clip_val=1,
                 smooth_end_epoch=-1,
                 use_ema=False,
                 load_ema=False,
                 ):
        super(VQAE, self).__init__()
        self.image_key = image_key
        encoder_ch_mult = eval(encoder_ch_mult) if isinstance(encoder_ch_mult, str) else encoder_ch_mult
        decoder_ch_mult = eval(decoder_ch_mult) if isinstance(decoder_ch_mult, str) else decoder_ch_mult
        cnn_param = dict(
            cnn_type=cnn_type,
            conv_in_out_2d=conv_in_out_2d,
            res_conv_2d=res_conv_2d,
            cnn_attention=cnn_attention,
            cnn_norm_axis=cnn_norm_axis,
            conv_inner_2d=conv_inner_2d,
        )
        self.encoder = Encoder(
            ch=base_ch,
            ch_mult=encoder_ch_mult,
            num_res_blocks=num_res_blocks,
            z_channels=embed_dim,
            patch_size=downsampled_size,
            temporal_patch_size=temporal_patch_size,
            cnn_param=cnn_param,
            use_checkpoint=use_checkpoint,
            use_vae=use_vae,
        )
        self.decoder = Decoder(
            ch=base_ch,
            ch_mult=decoder_ch_mult,
            num_res_blocks=num_res_blocks,
            z_channels=embed_dim,
            patch_size=downsampled_size,
            temporal_patch_size=temporal_patch_size,
            cnn_param=cnn_param,
            use_checkpoint=use_checkpoint,
            use_freq_dec=use_freq_dec,
            use_pxsf=use_pxsf
        )
        if lossconfig is not None:
            self.loss = instantiate_from_config(lossconfig)
            if self.loss.disc_type == 'dinodisc':
                self.loss.discriminator.dino_proxy = self.loss.discriminator.dino_proxy.to(self.device)
        self.v_patch_nums = eval(v_patch_nums) if isinstance(v_patch_nums, str) else v_patch_nums
        self.quantize = VQ2(n_embed, embed_dim, beta=0.25, remap=remap, sane_index_shape=sane_index_shape, v_patch_nums=self.v_patch_nums, smooth_end_epoch=smooth_end_epoch)

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys, load_ema=load_ema)

        self.use_ema = use_ema
        if self.use_ema:
            self.ema_model = deepcopy(self).to(self.device)
            for param in self.ema_model.parameters():
                param.requires_grad = False

        self.image_key = image_key
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor

        # calculate activated tokens
        self.num_codebook = len(self.v_patch_nums) if self.v_patch_nums is not None else 1
        self.register_buffer("total_visited_codes", torch.zeros((self.num_codebook, self.quantize.n_e), dtype=torch.long))
        # Important: This property activates manual optimization.
        self.automatic_optimization = False
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.gradient_clip_val = gradient_clip_val

    def init_from_ckpt(self, path, ignore_keys=list(), load_ema=False):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        if load_ema:
            print("Loading AE EMA model from checkpoint.")
            keys = list(sd.keys())
            for k in keys:
                if k.startswith("ema_model"):
                    # print("Deleting key {} from state_dict.".format(k))
                    sd[k.replace("ema_model.", "")] = sd[k]
                    del sd[k]
        else:
            print("Loading AE model from checkpoint without EMA.")
            keys = list(sd.keys())
            for k in keys:
                if k.startswith("ema_model"):
                    # print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    # print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(f"Restored AE from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")

    @torch.no_grad()
    def update_ema(self, decay=0.9999):
        """
        Step the EMA model towards the current model.
        """
        ema_params = OrderedDict(self.ema_model.named_parameters())
        model_params = OrderedDict(self.named_parameters())

        for name, param in model_params.items():
            if 'ema_model' in name:
                continue
            # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
            ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

    def load_ema(self):
        """
        Load the EMA model parameters into the current model.
        """
        ema_params = OrderedDict(self.ema_model.named_parameters())
        model_params = OrderedDict(self.named_parameters())
        for name, param in model_params.items():
            if 'ema_model' in name:
                continue
            param.data.copy_(ema_params[name].data)

    def encode(self, x):
        h = self.encoder(x)
        quant, emb_loss, info, vq_loss_dict = self.quantize(h, epoch=self.current_epoch)
        return quant, emb_loss, info, vq_loss_dict, h

    def decode(self, quant):
        dec = self.decoder(quant).clamp(-1, 1)
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input, return_pred_indices=False):
        quant, diff, (_,_,ind), vq_loss_dict, h = self.encode(input)
        dec = self.decode(quant)
        if return_pred_indices:
            return dec, diff, ind, vq_loss_dict, h
        return dec, diff, vq_loss_dict, h

    @maybe_compile
    def fhat_to_img(self, f_hat: torch.Tensor):
        return self.decode(f_hat)

    @maybe_compile
    def only_encode(self, x):
        """
        Encode the input image to the latent space.
        """
        f = self.encoder(x)
        return f

    @torch.no_grad()
    def img_to_nvg_input_output(self, x, predict_final=False):
        f = self.only_encode(x)
        return self.quantize.f_to_nvg_input_output(f, predict_final=predict_final)

    @torch.no_grad()
    def img_to_nvg_to_img(self, x, full_list=False, vis_labelmap=False, no_decode=False):
        nvg_input_output = self.img_to_nvg_input_output(x)
        nvg_output = nvg_input_output[:, :, :, -2]
        nvg_labelmap = nvg_input_output[:, :, :, -1]
        f_hat = self.quantize.nvg_output_to_fhat(nvg_output, full_list=full_list)
        if no_decode:
            return f_hat
        if full_list:
            if vis_labelmap:
                return [self.decode(f) for f in f_hat], nvg_labelmap
            else:
                return [self.decode(f) for f in f_hat]
        else:
            if vis_labelmap:
                return self.decode(f_hat), nvg_labelmap
            else:
                return self.decode(f_hat)


    @torch.no_grad()
    def nvg_output_to_img(self, nvg_output):
        f_hat = self.quantize.nvg_output_to_fhat(nvg_output)
        return self.fhat_to_img(f_hat)

    @torch.no_grad()
    def nvg_next_input(self, inp_i, out_i, stage):
        nvg_input = self.quantize.nvg_next_input(inp_i, out_i, stage)
        return nvg_input

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        return x.float()

    def training_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec, qloss, ind, vq_loss_dict, latent = self(x, return_pred_indices=True)
        bincounts = torch.stack([torch.bincount(c.flatten(), minlength=self.quantize.n_e) for c in ind])
        self.total_visited_codes[:len(bincounts)] += bincounts

        opt_ae, opt_disc = self.optimizers()

        # autoencode
        aeloss, log_dict = self.loss(qloss, x, xrec, 0, self.global_step, last_layer=self.get_last_layer(), split="train", latent=latent, enc_last_layer=self.get_enc_last_layer())
        vq_loss_dict = {f"train/{k}": vq_loss_dict[k] for k in vq_loss_dict}
        log_dict.update(vq_loss_dict)

        # discriminator
        discloss, log_dict_disc = self.loss(qloss, x, xrec, 1, self.global_step, last_layer=self.get_last_layer(), split="train", latent=latent, enc_last_layer=self.get_enc_last_layer())
        log_dict.update(log_dict_disc)
        self.log_dict(log_dict, prog_bar=False, logger=True, on_step=True, on_epoch=True)

        aeloss = aeloss / self.gradient_accumulation_steps
        discloss = discloss / self.gradient_accumulation_steps
        self.manual_backward(aeloss+discloss)
        if (batch_idx+1) % self.gradient_accumulation_steps == 0:
            self.toggle_optimizer(opt_ae)
            self.clip_gradients(opt_ae, self.gradient_clip_val, gradient_clip_algorithm='norm')
            opt_ae.step()
            opt_ae.zero_grad()
            self.untoggle_optimizer(opt_ae)
            self.toggle_optimizer(opt_disc)
            self.clip_gradients(opt_disc, self.gradient_clip_val, gradient_clip_algorithm='norm')
            opt_disc.step()
            opt_disc.zero_grad()
            self.untoggle_optimizer(opt_disc)
            if self.use_ema:
                self.update_ema()

    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec, qloss, ind, vq_loss_dict, latent = self(x, return_pred_indices=True)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0,
                                        self.global_step,
                                        last_layer=self.get_last_layer(),
                                        split="val",
                                        latent=latent,
                                        enc_last_layer=self.get_enc_last_layer(),
                                        )
        discloss, log_dict_disc = self.loss(qloss, x, xrec, 1,
                                            self.global_step,
                                            last_layer=self.get_last_layer(),
                                            split="val",
                                            latent=latent,
                                            enc_last_layer=self.get_enc_last_layer(),
                                            )
        log_dict_ae.update(log_dict_disc)
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_list = list(self.encoder.parameters())+list(self.decoder.parameters())+list(self.quantize.parameters())
        opt_ae = torch.optim.AdamW(opt_list, lr=lr, betas=(0.9, 0.95), weight_decay=0.0)
        opt_disc = torch.optim.AdamW(self.loss.discriminator.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.0005)
        scheduler_ae = torch.optim.lr_scheduler.CosineAnnealingLR(opt_ae, self.trainer.max_epochs)
        scheduler_disc = torch.optim.lr_scheduler.CosineAnnealingLR(opt_disc, self.trainer.max_epochs)
        return [opt_ae, opt_disc], [scheduler_ae, scheduler_disc]

    def get_last_layer(self):
        return self.decoder.conv_out.conv.weight

    def get_enc_last_layer(self):
        return self.encoder.conv_out.conv.weight

    def log_images(self, batch, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        xrec, _, _, _, _ = self(x)
        if x.shape[1] > 3:
            # colorize with random projection
            assert xrec.shape[1] > 3
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
        log["inputs"] = x
        log["reconstructions"] = xrec
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x

    def set_statisticsdir(self, statisticsdir):
        self.private_statsiticsdir = statisticsdir
        os.makedirs(self.private_statsiticsdir, exist_ok=True)

    @rank_zero_only
    def on_train_epoch_start(self):
        if self.current_epoch == 0:
            return
        os.makedirs(self.private_statsiticsdir, exist_ok=True)
        self.total_visited_codes = self.total_visited_codes.cpu()
        for i in range(self.total_visited_codes.shape[0]):
            plt.figure(figsize=(20, 3))
            plt.hist(self.total_visited_codes[i], bins=1000, color='blue', edgecolor='black', alpha=0.7)
            plt.title('Histogram of Activated Codebook')
            plt.xlabel('Activated Value')
            plt.ylabel('Frequency')
            plt.savefig(os.path.join(self.private_statsiticsdir, 'histogram_activated_codebook_epoch{}_{}.png'.format(self.current_epoch, i)))
            plt.close()
            plt.figure(figsize=(20, 3))
            plt.plot(self.total_visited_codes[i].numpy(), color='blue', alpha=0.7)
            plt.title('Line plot of  Activated Codebook of size: {}'.format(self.v_patch_nums[i]))
            plt.xlabel('Code Index')
            plt.ylabel('Activated Value')
            plt.savefig(os.path.join(self.private_statsiticsdir, 'lineplot_activated_codebook_epoch{}_{}.png'.format(self.current_epoch, i)))
            plt.close()
            if i == 0:
                current_total_visited_codes = self.total_visited_codes[i]
            else:
                current_total_visited_codes += self.total_visited_codes[i]
            utilization_rate = (self.total_visited_codes[i] > 0).float().mean()
            current_total_utilization_rate = (current_total_visited_codes > 0).float().mean()
            self.log("train/codebook_{}_utilization_rate".format(i), utilization_rate, logger=True, on_epoch=True, prog_bar=False)
            print('Epoch: {} Codebook {} utilization rate: {:.2f} total utilization rate: {:.2f}'.format(self.current_epoch, i, utilization_rate*100, current_total_utilization_rate*100))
        self.log("train/total_codebook_utilization_rate", current_total_utilization_rate, logger=True, on_epoch=True, prog_bar=False)
        self.total_visited_codes *= 0
        self.total_visited_codes = self.total_visited_codes.to(self.device)
