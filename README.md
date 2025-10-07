<p align="center">
<h1 align="center">Next Visual Granularity Generation</h1>
</p>
<p align="center">
<a href="https://arxiv.org/abs/2508.12811"><img alt='arXiv' src="https://img.shields.io/badge/arXiv-2508.12811-b31b1b.svg"></a>
<a href="https://yikai-wang.github.io/nvg/"><img alt='page' src="https://img.shields.io/badge/Project-Website-orange"></a>
</p>
<p align="center">
Yikai Wang, Zhouxia Wang, Zhonghua Wu, Qingyi Tao, Kang Liao, Chen Change Loy.<br>
S-Lab, Nanyang Technological University; SenseTime Research<br>
</p>

## Overview

We propose a novel approach to image generation by decomposing an image into a structured sequence, where each element in the sequence shares the same spatial resolution but differs in the number of unique tokens used, capturing different level of visual granularity.<br>
Image generation is carried out through our newly introduced Next Visual Granularity (NVG) generation framework, which generates a visual granularity sequence beginning from an empty image and progressively refines it, from global layout to fine details, in a structured manner. This iterative process encodes a hierarchical, layered representation that offers fine-grained control over the generation process across multiple granularity levels.<br>
We train a series of NVG models for class-conditional image generation on the ImageNet dataset and observe clear scaling behavior. Compared to the VAR series, NVG consistently outperforms it in terms of FID scores (3.30 → 3.03, 2.57 → 2.44, 2.09 → 2.06). We also conduct extensive analysis to showcase the capability and potential of the NVG framework. Our code and models will be released.<br>

## License

This repository, along with its code and models, is licensed under the S-Lab License 1.0. See [here](./LICENSE) for more details.


## Installation

### Create environment for NVG training and sampling

```bash
conda create -n nvg python=3.9
conda activate nvg
pip install -r requirements.txt
# only run the following command when you face issues related to libgl
conda install -c conda-forge libgl
```

### Evaluation environment ([OpenAI's evaluation tool](https://github.com/openai/guided-diffusion/tree/main/evaluations))

We recommend creating a separate conda environment named `oai-eva`:

```bash
python>=3.9
tensorflow-gpu>=2.0
scipy
requests
tqdm
```

## Checkpoints

Pretrained NVG models are available on [Hugging Face](https://huggingface.co/yikaiwang/nvg).
Please place all downloaded checkpoints in the **`ckpt/`** folder.

## Training and Evaluation

### Notes

We use pytorch compile mode for faster training and evaluation with a little extra cost at initialization. Ignore it via:
```
USE_TORCH_COMPILE=0
```
We also use WANDB to log training. Set **WANDB_API_KEY** in your environmnet, or use the offline mode
```
WANDB_MODE='offline'
```
*The training configs assume a single-machine-single-GPU setting, change it in the config file for your convenience.*

### Auto-Encoder

**Training**

```bash
torchrun --nnodes=1 --nproc_per_node=1 --rdzv_id=distributed_alldata --rdzv_backend=c10d --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT main.py -t True --base configs/downsample_configs/vq-f16-d32.yaml --enable_tf32 True
```

**Evaluation**

```bash
python eval_vq.py --config_file configs/downsample_configs/vq-f16-d32.yaml --ckpt_path ckpt/ae.ckpt
```

---

### Generator

**Training**

Configs for variants with different depths are named as d16, d20, d24, respectively. Change it for your convenience.
```bash
torchrun --nnodes=1 --nproc_per_node=1 --rdzv_id=distributed_alldata --rdzv_backend=c10d --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT main.py -t True --scale_lr --base configs/generator_configs/d16.yaml
```

**Sampling**

```bash
python eval_gen.py --eval_ema --batch_size 250 --config_file configs/generator_configs/d16.yaml --ckpt_path ckpt/d16.ckpt --content_cfg_scale=1-3.5 --structure_cfg_scale=1-2.5 --top_p=1-0.5 --samples_per_class=50 --structure_sampling_step=25 --sample_dir gen_results/d16
```

**Evaluation**

Download the reference batch from [here](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz) (provided by OpenAI) and save it in the **`ckpt/`** folder.

```bash
conda activate oai-eva
python metrics/evaluator.py ckpt/VIRTUAL_imagenet256_labeled.npz gen_results/d16.npz
```

## Acknowledgements

We thank the following open-sourcing projects:

[VAR](https://github.com/FoundationVision/VAR)<br>
[Taming-Transformers](https://github.com/CompVis/taming-transformers)<br>
[Infinity](https://github.com/FoundationVision/Infinity)<br>
[FLUX](https://github.com/black-forest-labs/flux)<br>
[SEED-Voken](https://github.com/TencentARC/SEED-Voken)<br>
