import torch
import numpy as np
from einops import rearrange
from torch import Tensor, nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from typing import List

ptdtype = {None: torch.float32, 'fp32': torch.float32, 'bf16': torch.bfloat16}


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, cnn_type="2d", causal_offset=0, temporal_down=False):
        super().__init__()
        self.cnn_type = cnn_type
        self.slice_seq_len = 17

        if cnn_type == "2d":
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        if cnn_type == "3d":
            if temporal_down == False:
                stride = (1, stride, stride)
            else:
                stride = (stride, stride, stride)
            self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=0)
            if isinstance(kernel_size, int):
                kernel_size = (kernel_size, kernel_size, kernel_size)
            self.padding = (
                kernel_size[0] - 1 + causal_offset,  # Temporal causal padding
                padding,  # Height padding
                padding  # Width padding
            )
        self.causal_offset = causal_offset
        self.stride = stride
        self.kernel_size = kernel_size

    def forward(self, x):
        if self.cnn_type == "2d":
            if x.ndim == 5:
                B, C, T, H, W = x.shape
                x = rearrange(x, "B C T H W -> (B T) C H W")
                x = self.conv(x)
                x = rearrange(x, "(B T) C H W -> B C T H W", T=T)
                return x
            else:
                return self.conv(x)
        if self.cnn_type == "3d":
            assert self.stride[0] == 1 or self.stride[0] == 2, f"only temporal stride = 1 or 2 are supported"
            xs = []
            for i in range(0, x.shape[2], self.slice_seq_len+self.stride[0]-1):
                st = i
                en = min(i+self.slice_seq_len, x.shape[2])
                _x = x[:,:,st:en,:,:]
                if i == 0:
                    _x = F.pad(_x, (self.padding[2], self.padding[2],  # Width
                            self.padding[1], self.padding[1],   # Height
                            self.padding[0], 0))                # Temporal
                else:
                    padding_0 = self.kernel_size[0] - 1
                    _x = F.pad(_x, (self.padding[2], self.padding[2],  # Width
                            self.padding[1], self.padding[1],   # Height
                            padding_0, 0))                      # Temporal
                    _x[:,:,:padding_0,
                        self.padding[1]:_x.shape[-2]-self.padding[1],
                        self.padding[2]:_x.shape[-1]-self.padding[2]] += x[:,:,i-padding_0:i,:,:]
                _x = self.conv(_x)
                xs.append(_x)
            try:
                x = torch.cat(xs, dim=2)
            except:
                device = x.device
                del x
                xs = [_x.cpu().pin_memory() for _x in xs]
                torch.cuda.empty_cache()
                x = torch.cat([_x.cpu() for _x in xs], dim=2).to(device=device)
            return x

class Normalize(nn.Module):
    def __init__(self, in_channels, norm_type, norm_axis="spatial"):
        super().__init__()
        self.norm_axis = norm_axis
        assert norm_type in ['group', 'batch', "no"]
        if norm_type == 'group':
            if in_channels % 32 == 0:
                self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
            elif in_channels % 24 == 0:
                self.norm = nn.GroupNorm(num_groups=24, num_channels=in_channels, eps=1e-6, affine=True)
            else:
                raise NotImplementedError
        elif norm_type == 'batch':
            self.norm = nn.SyncBatchNorm(in_channels, track_running_stats=False) # Runtime Error: grad inplace if set track_running_stats to True
        elif norm_type == 'no':
            self.norm = nn.Identity()

    def forward(self, x):
        if self.norm_axis == "spatial":
            if x.ndim == 4:
                x = self.norm(x)
            else:
                B, C, T, H, W = x.shape
                x = rearrange(x, "B C T H W -> (B T) C H W")
                x = self.norm(x)
                x = rearrange(x, "(B T) C H W -> B C T H W", T=T)
        elif self.norm_axis == "spatial-temporal":
            x = self.norm(x)
        else:
            raise NotImplementedError
        return x

def swish(x: Tensor) -> Tensor:
    try:
        return x * torch.sigmoid(x)
    except:
        device = x.device
        x = x.cpu().pin_memory()
        return (x*torch.sigmoid(x)).to(device=device)

class AttnBlock(nn.Module):
    def __init__(self, in_channels, norm_type='group', cnn_param=None):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels, norm_type, norm_axis=cnn_param["cnn_norm_axis"])

        self.q = Conv(in_channels, in_channels, kernel_size=1)
        self.k = Conv(in_channels, in_channels, kernel_size=1)
        self.v = Conv(in_channels, in_channels, kernel_size=1)
        self.proj_out = Conv(in_channels, in_channels, kernel_size=1)

    def attention(self, h_: Tensor) -> Tensor:
        B, _, T, _, _ = h_.shape
        h_ = self.norm(h_)
        h_ = rearrange(h_, "B C T H W -> (B T) C H W") # spatial attention only
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, h, w = q.shape
        q = rearrange(q, "b c h w -> b 1 (h w) c").contiguous()
        k = rearrange(k, "b c h w -> b 1 (h w) c").contiguous()
        v = rearrange(v, "b c h w -> b 1 (h w) c").contiguous()
        h_ = nn.functional.scaled_dot_product_attention(q, k, v)

        return rearrange(h_, "(b t) 1 (h w) c -> b c t h w", h=h, w=w, c=c, b=B, t=T)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.proj_out(self.attention(x))

class ResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, norm_type='group', cnn_param=None):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.norm1 = Normalize(in_channels, norm_type, norm_axis=cnn_param["cnn_norm_axis"])
        if cnn_param["res_conv_2d"] in ["half", "full"]:
            self.conv1 = Conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1, cnn_type="2d")
        else:
            self.conv1 = Conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1, cnn_type=cnn_param["cnn_type"])
        self.norm2 = Normalize(out_channels, norm_type, norm_axis=cnn_param["cnn_norm_axis"])
        if cnn_param["res_conv_2d"] in ["full"]:
            self.conv2 = Conv(out_channels, out_channels, kernel_size=3, stride=1, padding=1, cnn_type="2d")
        else:
            self.conv2 = Conv(out_channels, out_channels, kernel_size=3, stride=1, padding=1, cnn_type=cnn_param["cnn_type"])
        if self.in_channels != self.out_channels:
            self.nin_shortcut = Conv(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = swish(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = swish(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)

        return x + h

class Downsample(nn.Module):
    def __init__(self, in_channels, cnn_type="2d", spatial_down=False, temporal_down=False):
        super().__init__()
        assert spatial_down == True
        if cnn_type == "2d":
            self.pad = (0,1,0,1)
        if cnn_type == "3d":
            self.pad = (0,1,0,1,0,0) # add padding to the right for h-axis and w-axis. No padding for t-axis
        # no asymmetric padding in torch conv, must do it ourselves
        self.conv = Conv(in_channels, in_channels, kernel_size=3, stride=2, padding=0, cnn_type=cnn_type, temporal_down=temporal_down)

    def forward(self, x: Tensor):
        x = nn.functional.pad(x, self.pad, mode="constant", value=0)
        x = self.conv(x)
        return x

class Upsample(nn.Module):
    def __init__(self, in_channels, cnn_type="2d", spatial_up=False, temporal_up=False, use_pxsl=False):
        super().__init__()
        if cnn_type == "2d":
            self.scale_factor = 2
            self.causal_offset = 0
        else:
            assert spatial_up == True
            if temporal_up:
                self.scale_factor = (2,2,2)
                self.causal_offset = -1
            else:
                self.scale_factor = (1,2,2)
                self.causal_offset = 0
        self.use_pxsl = use_pxsl
        if self.use_pxsl:
            self.conv = Conv(in_channels, in_channels*4, kernel_size=3, stride=1, padding=1, cnn_type=cnn_type, causal_offset=self.causal_offset)
            self.pxsl = nn.PixelShuffle(2)
        else:
            self.conv = Conv(in_channels, in_channels, kernel_size=3, stride=1, padding=1, cnn_type=cnn_type, causal_offset=self.causal_offset)

    def forward(self, x: Tensor):
        if self.use_pxsl:
            x = self.conv(x)
            x = self.pxsl(x)
        else:
            try:
                x = F.interpolate(x, scale_factor=self.scale_factor, mode="nearest")
            except:
                # shard across channel
                _xs = []
                for i in range(x.shape[1]):
                    _x = F.interpolate(x[:,i:i+1,...], scale_factor=self.scale_factor, mode="nearest")
                    _xs.append(_x)
                x = torch.cat(_xs, dim=1)
            x = self.conv(x)
        return x

class Encoder(nn.Module):
    def __init__(
        self,
        ch: int,
        ch_mult: List[int],
        num_res_blocks: int,
        z_channels: int,
        in_channels = 3,
        patch_size=8, temporal_patch_size=4,
        norm_type='group', cnn_param=None,
        use_checkpoint=False,
        use_vae=True,
    ):
        super().__init__()
        self.max_down = np.log2(patch_size)
        self.temporal_max_down = np.log2(temporal_patch_size)
        self.temporal_down_offset = self.max_down - self.temporal_max_down
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.in_channels = in_channels
        self.cnn_param = cnn_param
        self.use_checkpoint = use_checkpoint
        if cnn_param["conv_in_out_2d"] == "yes": # "yes" for video
            self.conv_in = Conv(in_channels, ch, kernel_size=3, stride=1, padding=1, cnn_type="2d")
        else:
            self.conv_in = Conv(in_channels, ch, kernel_size=3, stride=1, padding=1, cnn_type=cnn_param["cnn_type"])

        in_ch_mult = (1,) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        block_in = self.ch
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, norm_type=norm_type, cnn_param=cnn_param))
                block_in = block_out
            down = nn.Module()
            down.block = block
            down.attn = attn
            # downsample, stride=1, stride=2, stride=2 for 4x8x8 Video VAE
            spatial_down = True if i_level < self.max_down else False
            temporal_down = True if i_level < self.max_down and i_level >= self.temporal_down_offset else False
            if spatial_down or temporal_down:
                down.downsample = Downsample(block_in, cnn_type=cnn_param["cnn_type"], spatial_down=spatial_down, temporal_down=temporal_down)
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, norm_type=norm_type, cnn_param=cnn_param)
        if cnn_param["cnn_attention"] == "yes":
            self.mid.attn_1 = AttnBlock(block_in, norm_type, cnn_param=cnn_param)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, norm_type=norm_type, cnn_param=cnn_param)

        # end
        self.norm_out = Normalize(block_in, norm_type, norm_axis=cnn_param["cnn_norm_axis"])
        if cnn_param["conv_inner_2d"] == "yes":
            self.conv_out = Conv(block_in, (int(use_vae) + 1) * z_channels, kernel_size=3, stride=1, padding=1, cnn_type="2d")
        else:
            self.conv_out = Conv(block_in, (int(use_vae) + 1) * z_channels, kernel_size=3, stride=1, padding=1, cnn_type=cnn_param["cnn_type"])

    def forward(self, x, return_hidden=False):
        if not self.use_checkpoint:
            return self._forward(x, return_hidden=return_hidden)
        else:
            return checkpoint.checkpoint(self._forward, x, return_hidden, use_reentrant=False)

    def _forward(self, x: Tensor, return_hidden=False) -> Tensor:
        # downsampling
        h0 = self.conv_in(x)
        hs = [h0]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1])
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if hasattr(self.down[i_level], "downsample"):
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        hs_mid = [h]
        h = self.mid.block_1(h)
        if self.cnn_param["cnn_attention"] == "yes":
            h = self.mid.attn_1(h)
        h = self.mid.block_2(h)
        hs_mid.append(h)
        # end
        h = self.norm_out(h)
        h = swish(h)
        h = self.conv_out(h)
        if return_hidden:
            return h, hs, hs_mid
        else:
            return h


class Decoder(nn.Module):
    def __init__(
        self,
        ch: int,
        ch_mult: List[int],
        num_res_blocks: int,
        z_channels: int,
        out_ch = 3,
        patch_size=8, temporal_patch_size=4,
        norm_type="group", cnn_param=None,
        use_checkpoint=False,
        use_freq_dec=False, # use frequency features for decoder
        use_pxsf=False
    ):
        super().__init__()
        self.max_up = np.log2(patch_size)
        self.temporal_max_up = np.log2(temporal_patch_size)
        self.temporal_up_offset = self.max_up - self.temporal_max_up
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.ffactor = 2 ** (self.num_resolutions - 1)
        self.cnn_param = cnn_param
        self.use_checkpoint = use_checkpoint
        self.use_freq_dec = use_freq_dec
        self.use_pxsf = use_pxsf

        # compute in_ch_mult, block_in and curr_res at lowest res
        block_in = ch * ch_mult[self.num_resolutions - 1]

        # z to block_in
        if cnn_param["conv_inner_2d"] == "yes":
            self.conv_in = Conv(z_channels, block_in, kernel_size=3, stride=1, padding=1, cnn_type="2d")
        else:
            self.conv_in = Conv(z_channels, block_in, kernel_size=3, stride=1, padding=1, cnn_type=cnn_param["cnn_type"])

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, norm_type=norm_type, cnn_param=cnn_param)
        if cnn_param["cnn_attention"] == "yes":
            self.mid.attn_1 = AttnBlock(block_in, norm_type=norm_type, cnn_param=cnn_param)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, norm_type=norm_type, cnn_param=cnn_param)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, norm_type=norm_type, cnn_param=cnn_param))
                block_in = block_out
            up = nn.Module()
            up.block = block
            up.attn = attn
            # upsample, stride=1, stride=2, stride=2 for 4x8x8 Video VAE, offset 1 compared with encoder
            # https://github.com/black-forest-labs/flux/blob/b4f689aaccd40de93429865793e84a734f4a6254/nvg/flux/modules/autoencoder.py#L228
            spatial_up = True if 1 <= i_level <= self.max_up else False
            temporal_up = True if 1 <= i_level <= self.max_up and i_level >= self.temporal_up_offset+1 else False
            if spatial_up or temporal_up:
                up.upsample = Upsample(block_in, cnn_type=cnn_param["cnn_type"], spatial_up=spatial_up, temporal_up=temporal_up, use_pxsl=self.use_pxsf)
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in, norm_type, norm_axis=cnn_param["cnn_norm_axis"])
        if cnn_param["conv_in_out_2d"] == "yes":
            self.conv_out = Conv(block_in, out_ch, kernel_size=3, stride=1, padding=1, cnn_type="2d")
        else:
            self.conv_out = Conv(block_in, out_ch, kernel_size=3, stride=1, padding=1, cnn_type=cnn_param["cnn_type"])

    def forward(self, z):
        if not self.use_checkpoint:
            return self._forward(z)
        else:
            return checkpoint.checkpoint(self._forward, z, use_reentrant=False)

    def _forward(self, z: Tensor) -> Tensor:
        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h)
        if self.cnn_param["cnn_attention"] == "yes":
            h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if hasattr(self.up[i_level], "upsample"):
                h = self.up[i_level].upsample(h)

        # end
        h = self.norm_out(h)
        h = swish(h)
        h = self.conv_out(h)
        return h
