"""
We provide Tokenizer Evaluation code here.
Refer to
https://github.com/richzhang/PerceptualSimilarity
https://github.com/mseitzer/pytorch-fid
"""

import torch
import os

from omegaconf import OmegaConf
import yaml
import numpy as np
from PIL import Image
from tqdm import tqdm
from pytorch_lightning import seed_everything

from nvg.models.downsample import VQAE
from main import instantiate_from_config

from metrics.inception import InceptionV3, calculate_frechet_distance, calculate_frechet_distance_1
import lpips
from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from skimage.metrics import structural_similarity as ssim_loss
import argparse

import warnings
warnings.filterwarnings("ignore")

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_config(config_path, display=False):
    config = OmegaConf.load(config_path)
    if display:
        print(yaml.dump(OmegaConf.to_container(config)))
    return config

def load_vq(config, ckpt_path=None, is_gumbel=False):
    model = VQAE(**config.model.params)
    del model.loss
    del model.ema_model
    del model.total_visited_codes
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"missing keys: {missing} unexpected keys: {unexpected}")
    return model.eval()

def custom_to_pil(x):
    x = x.detach().cpu()
    x = x.permute(1,2,0).numpy()
    x = (255*x).astype(np.uint8)
    x = Image.fromarray(x)
    if not x.mode == "RGB":
        x = x.convert("RGB")
    return x

def get_args():
    parser = argparse.ArgumentParser(description="inference parameters")
    parser.add_argument("--config_file", required=True, type=str)
    parser.add_argument("--ckpt_path", required=True, type=str)
    parser.add_argument("--image_size", default=256, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--eval_ema", action="store_true", help="use ema model")
    parser.add_argument("--return_indices", action="store_true", help="return indices")
    parser.add_argument("--full_list", action="store_true", help="return full list")
    parser.add_argument("--vis_labelmap", action="store_true", help="visualize labelmap")
    return parser.parse_args()

def main(args):
    seed_everything(42)

    config = load_config(args.config_file, display=False)
    model = load_vq(config, ckpt_path=args.ckpt_path).to(DEVICE)
    if args.eval_ema:
        model.load_ema()
    num_codebook = len(model.v_patch_nums)

    if args.return_indices:
        usage = torch.zeros((num_codebook, model.quantize.n_e), dtype=torch.long, device=DEVICE)


    # FID score related
    dims = 2048
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    inception_model = InceptionV3([block_idx], normalize_input=False).float().to(DEVICE)
    inception_model.eval()

    config_data = config.data
    config_data.params.batch_size = args.batch_size
    config_data.params.validation.params.config.size = args.image_size
    del config_data.params.train
    dataset = instantiate_from_config(config_data)
    dataset.prepare_data()
    dataset.setup()
    pred_xs = []
    pred_recs = []

    # LPIPS score related
    loss_fn_alex = lpips.LPIPS(net='alex').to(DEVICE)
    loss_fn_vgg = lpips.LPIPS(net='vgg').to(DEVICE)
    lpips_alex = 0.0
    lpips_vgg = 0.0

    # SSIM score related
    ssim_value = 0.0

    # PSNR score related
    psnr_value = 0.0

    num_images = 0
    vis_dir = "recon_vis"
    if args.vis_labelmap:
        os.makedirs(os.path.join(vis_dir), exist_ok=True)
    with torch.no_grad():
        for idx, batch in tqdm(enumerate(dataset._val_dataloader()), ncols=0):
            images = batch["image"].permute(0, 3, 1, 2).to(DEVICE).float()
            num_images += images.shape[0]

            reconstructed_images = model.img_to_nvg_to_img(images, full_list=args.full_list, vis_labelmap=args.vis_labelmap)

            if args.return_indices:
                reconstructed_images, indices = reconstructed_images

            reconstructed_images = (reconstructed_images.detach() + 1) * 127.5
            reconstructed_images = reconstructed_images.to(torch.uint8).clamp(0, 255).float() / 127.5 - 1

            if args.return_indices:
                bincounts = torch.stack([torch.bincount(c.flatten(), minlength=model.quantize.n_e) for c in indices])
                usage += bincounts

            # calculate lpips
            lpips_alex += loss_fn_alex(images, reconstructed_images).sum()
            lpips_vgg += loss_fn_vgg(images, reconstructed_images).sum()

            # calculate fid
            pred_x = inception_model(images)[0]
            pred_x = pred_x.squeeze(3).squeeze(2).cpu().numpy()
            pred_rec = inception_model(reconstructed_images)[0]
            pred_rec = pred_rec.squeeze(3).squeeze(2).cpu().numpy()

            pred_xs.append(pred_x)
            pred_recs.append(pred_rec)

            images = (images + 1) / 2
            reconstructed_images = (reconstructed_images + 1) / 2

            #calculate PSNR and SSIM
            rgb_restored = (reconstructed_images * 255.0).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
            rgb_gt = (images * 255.0).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
            rgb_restored = rgb_restored.astype(np.float32) / 255.
            rgb_gt = rgb_gt.astype(np.float32) / 255.
            B, _, _, _ = rgb_restored.shape
            for i in range(B):
                rgb_restored_s, rgb_gt_s = rgb_restored[i], rgb_gt[i]
                ssim_value += ssim_loss(rgb_restored_s, rgb_gt_s, data_range=1.0, channel_axis=-1)
                psnr_value += psnr_loss(rgb_gt_s, rgb_restored_s)

    pred_xs = np.concatenate(pred_xs, axis=0)
    pred_recs = np.concatenate(pred_recs, axis=0)

    mu_x = np.mean(pred_xs, axis=0)
    sigma_x = np.cov(pred_xs, rowvar=False)
    mu_rec = np.mean(pred_recs, axis=0)
    sigma_rec = np.cov(pred_recs, rowvar=False)


    fid_value = calculate_frechet_distance(mu_x, sigma_x, mu_rec, sigma_rec)
    fid_value_1 = calculate_frechet_distance_1(mu_x, sigma_x, mu_rec, sigma_rec)
    lpips_alex_value = lpips_alex / num_images
    lpips_vgg_value = lpips_vgg / num_images
    ssim_value = ssim_value / num_images
    psnr_value = psnr_value / num_images

    if args.return_indices:
        utilization = [torch.mean((u > 0).float()).item() for u in usage]

    print("FID: ", fid_value)
    print("FID_1: ", fid_value_1)
    print("LPIPS_ALEX: ", lpips_alex_value.item())
    print("LPIPS_VGG: ", lpips_vgg_value.item())
    print("SSIM: ", ssim_value)
    print("PSNR: ", psnr_value)
    if args.return_indices:
        for i, u in enumerate(utilization):
            print(f"utilization_{i}", u)

if __name__ == "__main__":
    args = get_args()
    main(args)