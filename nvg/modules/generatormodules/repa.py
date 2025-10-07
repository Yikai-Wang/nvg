import torch
import timm
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.transforms import Normalize

@torch.no_grad()
def load_encoders(enc_name='dinov2-vit-b', resolution=256):
    assert (resolution == 256) or (resolution == 512)

    encoder_type, architecture, model_config = enc_name.split('-')
    # Currently, we only support 512x512 experiments with DINOv2 encoders.
    if resolution == 512:
        if encoder_type != 'dinov2':
            raise NotImplementedError(
                "Currently, we only support 512x512 experiments with DINOv2 encoders."
                )

    if 'reg' in encoder_type:
        encoder = torch.hub.load('facebookresearch/dinov2', f'dinov2_vit{model_config}14_reg')
    else:
        encoder = torch.hub.load('facebookresearch/dinov2', f'dinov2_vit{model_config}14')
    del encoder.head
    patch_resolution = 16 * (resolution // 256)
    encoder.pos_embed.data = timm.layers.pos_embed.resample_abs_pos_embed(
        encoder.pos_embed.data, [patch_resolution, patch_resolution],
    )
    encoder.head = torch.nn.Identity()
    encoder.eval()

    return encoder

def preprocess_raw_image(x):
    resolution = x.shape[-1]
    x = (x + 1) / 2
    x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
    x = torch.nn.functional.interpolate(x, 224 * (resolution // 256), mode='bicubic')
    return x