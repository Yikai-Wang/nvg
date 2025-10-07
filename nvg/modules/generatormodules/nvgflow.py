import torch
import os
import math
from torch import Tensor, nn
import torch.nn.functional as F
from einops import rearrange
from typing import Callable, List

from .layers import (
    SelfAttnBlock,
    LastLayer,
    LastLinearLayer,
    EmbedND,
    MLPEmbedder,
    build_mlp,
    timestep_embedding,
)


compile_mode = os.getenv("USE_TORCH_COMPILE", "1") == "1"
print("NVG Compile mode:", compile_mode)

def maybe_compile(fn):
    if compile_mode:
        return torch.compile(fn)
    else:
        return fn

def time_shift(mu: float, sigma: float, t: Tensor):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)

def get_lin_function(
    x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15
) -> Callable[[float], float]:
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b

def get_schedule(
    num_steps: int,
    image_seq_len: int,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
    shift: bool = True,
) -> List[float]:
    # extra step for zero
    timesteps = torch.linspace(1, 0, num_steps + 1)

    # shifting the schedule to favor high timesteps for higher signal images
    if shift:
        # estimate mu based on linear estimation between two points
        mu = get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
        timesteps = time_shift(mu, 1.0, timesteps)

    return timesteps.tolist()

def gumbel_topk(probs, k, random=True):
    probs = probs.clamp(min=-1, max=1) + 1 # ensure all probs are positive
    probs = probs / probs.sum(dim=-1, keepdim=True)
    if random:
        gumbel = -torch.log(-torch.log(torch.rand_like(probs) + 1e-8) + 1e-8)
        scores = torch.log(probs + 1e-8) + gumbel
    else:
        scores = probs
    topk = torch.topk(scores, k=k, dim=-1).indices
    binary = torch.zeros_like(probs, dtype=torch.long)
    binary.scatter_(-1, topk, 1)
    return binary

### from https://huggingface.co/transformers/v3.2.0/_modules/transformers/generation_utils.html
def top_k_top_p_filtering(
    logits,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
):
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
        if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        Make sure we keep at least min_tokens_to_keep per batch example in the output
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits

def sample(logits, temperature: float=1.0, top_k: int=0, top_p: float=1.0, sample_logits=True):
    # modified from https://github.com/FoundationVision/LlamaGen/autoregressive/models/generate.py
    if not sample_logits:
        probs = F.softmax(logits, dim=-1)
        idx = probs.argmax(dim=-1)
        return idx, probs
    logits = logits / max(temperature, 1e-5)
    if top_k > 0 or top_p < 1.0:
        logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
    probs = F.softmax(logits, dim=-1)
    batch_size = logits.shape[0]
    probs = rearrange(probs, "b l c -> (b l) c")
    if sample_logits:
        idx = torch.multinomial(probs, num_samples=1)
    else:
        _, idx = torch.topk(probs, k=1, dim=-1)
    idx = rearrange(idx, "(b l) c -> b (l c)", b=batch_size)
    probs = rearrange(probs, "(b l) c -> b l c", b=batch_size)
    return idx, probs

class NVGformer(nn.Module):
    def __init__(self,
                 vocab_size,
                 num_tokens,
                 mlp_ratio,
                 depth,
                 drop_path=False,
                 use_repa=False,
                 disable_vec_head=False,
                 img_input_channel=32,
                 num_classes=1000,
                 structure_down_factor=4,
                 head_dim=64,
                 ):
        super().__init__()
        self.hidden_size = depth * 64
        self.head_dim = head_dim
        if self.head_dim == 64:
            self.num_heads = depth
        elif self.head_dim == 128:
            self.num_heads = depth // 2
        else:
            raise ValueError("head_dim must be 64 or 128, got {}".format(self.head_dim))
        self.depth = depth

        if drop_path:
            drop_path_rate = 0.1 * depth/24
            self.drop_path_rate = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        else:
            self.drop_path_rate = [0] * depth

        self.num_tokens = tuple(num_tokens)
        self.vocab_size = vocab_size
        self.num_step = len(self.num_tokens)
        self.mlp_ratio = mlp_ratio

        self.use_repa = use_repa

        self.img_input_channel = img_input_channel
        self.num_classes = num_classes

        self.disable_vec_head = disable_vec_head

        # content part
        self.content_start_embed = nn.Parameter(torch.randn(1, self.num_tokens[-1], self.img_input_channel))
        self.content_embed = nn.Linear(self.img_input_channel, self.hidden_size, bias=True)
        self.content_class_embed = nn.Embedding(self.num_classes+1, self.hidden_size)
        self.content_stage_embed = nn.Embedding(self.num_step, self.hidden_size)
        self.content_blocks = nn.ModuleList([SelfAttnBlock(
            self.hidden_size,
            self.num_heads,
            mlp_ratio=self.mlp_ratio,
            drop_path_p=self.drop_path_rate[block_idx],
            ) for block_idx in range(self.depth)])

        self.x0_head = LastLayer(self.hidden_size, out_size=self.img_input_channel)

        if self.disable_vec_head:
            self.cls_head = LastLinearLayer(self.img_input_channel, self.vocab_size)
        else:
            self.cls_head = LastLayer(self.img_input_channel, mod_size=self.hidden_size, out_size=self.vocab_size)

        # structure part
        self.structure_down_factor = structure_down_factor
        self.init_structure_predictor(down_factor=self.structure_down_factor)

        self.init_rope_ids()
        self.initialize_weights()
        self.init_content_scale_factors()

        self.print_model_info()
        if compile_mode:
            print("Compiling NVGformer heads...")
            # compile_args = {"mode": "reduce-overhead", "fullgraph": True, "dynamic": False, "backend": "inductor"}
            compile_args = {}
            self.compile_heads(arg=compile_args)

    def compile_heads(self, arg=None):
        self.content_embed.compile(**arg)
        self.structure_embed.compile(**arg)

    def print_model_info(self):
        content_model_size = 0
        structure_model_size = 0

        content_model_size = self.content_start_embed.numel()
        content_model_size = sum(p.numel() for p in self.content_blocks.parameters() if p.requires_grad)
        content_model_size += sum(p.numel() for p in self.content_embed.parameters() if p.requires_grad)
        content_model_size += sum(p.numel() for p in self.content_class_embed.parameters() if p.requires_grad)
        content_model_size += sum(p.numel() for p in self.content_stage_embed.parameters() if p.requires_grad)
        content_model_size += sum(p.numel() for p in self.x0_head.parameters() if p.requires_grad)
        content_model_size += sum(p.numel() for p in self.cls_head.parameters() if p.requires_grad)

        structure_model_size = sum(p.numel() for p in self.structure_blocks.parameters() if p.requires_grad)
        structure_model_size += sum(p.numel() for p in self.structure_embed.parameters() if p.requires_grad)
        structure_model_size += sum(p.numel() for p in self.structure_class_embed.parameters() if p.requires_grad)
        structure_model_size += sum(p.numel() for p in self.structure_content_cond_embed.parameters() if p.requires_grad)
        structure_model_size += sum(p.numel() for p in self.structure_stage_embed.parameters() if p.requires_grad)
        structure_model_size += sum(p.numel() for p in self.structure_time_embed.parameters() if p.requires_grad)
        structure_model_size += sum(p.numel() for p in self.structure_head.parameters() if p.requires_grad)
        total_model_size = content_model_size + structure_model_size
        print("NVGformer Model Info:")
        print("  - Content Model Size: {:.2f} M".format(content_model_size / 1e6))
        print("  - Structure Model Size: {:.2f} M".format(structure_model_size / 1e6))
        print("  - Total Model Size: {:.2f} M".format(total_model_size / 1e6))

    def init_structure_predictor(self, down_factor=1):
        self.structure_embed = nn.Linear(8, self.hidden_size//down_factor)
        self.structure_class_embed = nn.Embedding(self.num_classes+1, self.hidden_size//down_factor)
        self.structure_content_cond_embed = nn.Linear(self.img_input_channel, self.hidden_size//down_factor)
        self.structure_stage_embed = nn.Embedding(self.num_step-2, self.hidden_size//down_factor)
        self.structure_time_embed = MLPEmbedder(in_dim=64, hidden_dim=self.hidden_size//down_factor) # ablate
        self.structure_blocks = nn.ModuleList([SelfAttnBlock(
            self.hidden_size//down_factor,
            self.num_heads//down_factor,
            mlp_ratio=self.mlp_ratio,
            drop_path_p=self.drop_path_rate[block_idx],
            ) for block_idx in range(self.depth)])
        self.structure_head = LastLayer(self.hidden_size//down_factor, out_size=8)

    def init_rope_ids(self):
        if self.head_dim == 64:
            axes_dim = [8] + [2] * (self.num_step - 1) + [20] * 2 # txt, structure, img
        elif self.head_dim == 96:
            axes_dim = [12] + [3] * (self.num_step - 1) + [30] * 2
        elif self.head_dim == 128:
            axes_dim = [16] + [4] * (self.num_step - 1) + [40] * 2 # txt, structure, img
        self.rope_embedder = EmbedND(dim=64, theta=10_000, axes_dim=axes_dim)
        L = self.num_tokens[-1]
        h = w = int(L**0.5)
        img_ids = torch.zeros(h, w, len(axes_dim), dtype=torch.long)
        img_ids[..., -2] = img_ids[..., -2] + torch.arange(h)[:, None]
        img_ids[..., -1] = img_ids[..., -1] + torch.arange(w)[None, :]
        img_ids = rearrange(img_ids, "h w c -> (h w) c")
        txt_ids = torch.zeros(1, len(axes_dim))
        txt_ids[0] = 1
        ids = torch.cat((txt_ids, img_ids), dim=0).unsqueeze(0)
        self.register_buffer("rope_ids", ids)

    def initialize_weights(self):
        torch.nn.init.normal_(self.content_class_embed.weight, std=.02)
        torch.nn.init.normal_(self.content_stage_embed.weight, std=.02)
        torch.nn.init.normal_(self.structure_class_embed.weight, std=.02)
        torch.nn.init.normal_(self.structure_stage_embed.weight, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    def init_content_scale_factors(self):
        # normalize factors for each step
        scale_factors = torch.tensor([1.0, 4.1993, 3.4706, 3.0520, 2.7430, 2.4842, 2.2407, 2.0088, 1.7954, 1.6080])
        self.register_buffer("scale_factors", scale_factors)

    def set_repa_alignment_layer(self, repa_depth, repa_dims):
        self.repa_depth = repa_depth
        self.projector = build_mlp(self.hidden_size, 2048, repa_dims)
        print("Use Layer-{} to perform REPA alignment".format(repa_depth))
        projector_size = sum(p.numel() for p in self.projector.parameters() if p.requires_grad)
        print("Projector Model Size: {} M".format(projector_size/1e6))

    def structure_int_to_feat(self, structure: Tensor, stage) -> Tensor:
        """Convert structure to binary feature map.
        Args:
            structure: Tensor of shape (B, L) where L is the number of tokens.
            stage: Current stage of the model.
        Returns:
            Tensor of shape (B, L, num_step-1) where num_step is the number of steps.
            -1 since 2^(9-1) = 256 classes.
        """
        if stage == 0:
            structure_feat = torch.ones(structure.shape[0], structure.shape[1], self.num_step-1, dtype=torch.uint8, device=structure.device)
        else:
            bits = ((structure.unsqueeze(-1) >> torch.arange(stage-1, -1, -1, device=structure.device))) & 1
            bits = bits.to(torch.uint8) * 2
            structure_feat = torch.ones(structure.shape[0], structure.shape[1], self.num_step-1, dtype=bits.dtype, device=structure.device)
            structure_feat[:, :, :stage] = bits
        return structure_feat

    def get_compressed_logits(self, logits, structure, stage):
        """Get compressed logits for the current stage.
        Args:
            logits: Tensor of shape (B, L, C) where L is the number of tokens and C is the number of classes.
            structure: Tensor of shape (B, L) where L is the number of tokens.
            stage: Current stage of the model.
        Returns:
            Tensor of shape (B, N, C) where N is the number of unique tokens for the current stage.
        """
        labelmap_one_hot = torch.nn.functional.one_hot(structure, num_classes=self.num_tokens[stage]).float()
        labelmap_one_hot = rearrange(labelmap_one_hot, "B L N -> B N L")
        logits_sum = torch.bmm(labelmap_one_hot, logits)
        class_counts = labelmap_one_hot.sum(dim=-1, keepdim=True).clamp(min=1)
        logits = logits_sum / class_counts
        return logits

    def scale_content_inference(self, img, stage, batch_size):
        img = img * self.scale_factors[stage]
        z = self.content_start_embed.expand(batch_size, -1, -1) if stage == 0 else img
        return z

    def prepare_inference_inputs_content(self, img, structure, txt, batch_size, stage):
        """Prepare inputs for the content part of the model.
        Args:
            img: Tensor of shape (B, L, C) where B is the batch size, L is the number of tokens, and C is the number of channels.
            structure: Tensor of shape (B, L) where L is the number of tokens.
            txt: class idx.
            batch_size: Batch size.
            stage: Current stage of the model.
            epsilon: Start noise.
        Returns:
            z: Noised image.
            vec: Context vector.
            s: Structure feature map.
        """
        context = self.content_class_embed(txt)
        content = self.scale_content_inference(img, stage=stage, batch_size=batch_size)
        content = self.content_embed(content)
        z = torch.cat([context, content], dim=1)
        vec = context.squeeze(1) + self.content_stage_embed.weight[stage:stage+1]
        s = self.structure_int_to_feat(structure, stage).to(z.dtype)
        return z, vec, s

    def scale_content_train(self, img, batch_size=None):
        scale = rearrange(self.scale_factors[:-1], "s -> 1 s 1 1").expand(batch_size, -1, img.shape[-2], img.shape[-1])
        img = img * scale
        z = torch.cat([self.content_start_embed.unsqueeze(0).expand(batch_size, -1, -1, -1), img[:, 1:]], dim=1)
        return z

    def prepare_train_inputs_content(self, img, structure, txt, batch_size):
        """Prepare inputs for the content part of the model.
        Args:
            img: Tensor of shape (B, S, L, C) where B is the batch size, S is the number of stages, L is the number of tokens, and C is the number of channels.
            structure: Tensor of shape (B, S, L) where S is the number of stages and L is the number of tokens.
            txt: class idx.
            batch_size: Batch size.
        Returns:
            z: image.
            vec: Context vector.
            s: Structure feature map.
        """
        z = self.scale_content_train(img, batch_size=batch_size)
        z = self.content_embed(z)
        time_embed = self.content_stage_embed.weight.unsqueeze(0).expand(batch_size, -1, -1)
        context = self.content_class_embed(txt)
        vec = context + time_embed
        vec = rearrange(vec, "b s c -> (b s) c")
        context = context.unsqueeze(1).repeat(1, self.num_step, 1, 1)
        z = torch.cat([context, z], dim=2)
        z = rearrange(z, "b s l c -> (b s) l c")
        s = torch.stack([self.structure_int_to_feat(structure[:, stage], stage) for stage in range(self.num_step)], dim=1)
        s = rearrange(s, "b s l c-> (b s) l c").to(z.dtype)
        return z, vec, s

    def cluster(self, current_structure, next_structure_logits, stage, use_gumbel_topk=False):
        """Cluster the current structure to get the next structure.
        Args:
            current_structure: Tensor of shape (B, L) where B is the batch size and L is the number of tokens. Values are in [0, 2^stage-1].
            next_structure_logits: Tensor of shape (B, L) where values are the predictions for the next structure.
            stage: Current stage of the model.
        Returns:
            next_structure: Tensor of shape (B, L) where B is the batch size and L is the number of tokens. Values are in [0, 2^stage].
            We follow the structure of dividing class j to class 2*j and 2*j+1.
        """
        next_structure = torch.zeros_like(current_structure)
        classes = self.num_tokens[stage]
        num_per_next_class = self.num_tokens[-1] // self.num_tokens[stage+1]
        locations_sorted = torch.argsort(current_structure, dim=-1, stable=True)
        logits_sorted = torch.gather(next_structure_logits, 1, locations_sorted)
        logits_sorted = rearrange(logits_sorted, "B (C E) -> B C E", C=classes)
        new_classes = (torch.arange(classes).unsqueeze(0) * 2).to(current_structure.device)
        binary_samples = gumbel_topk(logits_sorted, num_per_next_class, random=use_gumbel_topk)
        new_tokens = new_classes.unsqueeze(-1) + binary_samples
        new_tokens = rearrange(new_tokens, "B C E -> B (C E)")
        next_structure.scatter_(1, locations_sorted, new_tokens)
        return next_structure

    @torch.no_grad()
    def structure_sampling(self, epsilon, content, txt, s, structure, stage, rope, use_cfg, cfg_scale, sampling_step=50, use_gumbel_topk=False):
        """Sample the structure for the current stage.
        Args:
            epsilon: start noise.
            content: Canvas of stage i, used to predict the structure of stage i+1.
            txt: class idx.
            s: Structure feature map.
            structure: Structure of stage i.
            stage: Current stage of the model.
            rope: Structure RoPE embedding.
        Returns:
            structure_pred: Tensor of shape (B, L) where B is the batch size and L is the number of tokens.
        """
        if stage == self.num_step - 2:
            # each parent will have only 2 tokens, so the order is not important
            structure_pred = self.cluster(structure, epsilon[:, :, stage], stage)
            return structure_pred
        context = self.structure_class_embed(txt) # B, C
        if use_cfg:
            un_txt = self.num_classes * torch.ones_like(txt)
            un_context = self.structure_class_embed(un_txt)
        content = self.structure_content_cond_embed(content) # B, L, C
        s = s - 1 # [-1, 1]
        z = s.clone()
        z[:, :, stage:] = epsilon[:, :, stage:]
        B = z.shape[0]
        stage_embed = self.structure_stage_embed.weight[stage:stage+1].expand(B, -1, -1)
        timesteps = get_schedule(sampling_step, self.num_tokens[-1])
        if use_cfg:
            rope = torch.cat((rope, rope), dim=0)
        for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
            t = torch.full((B, 1), t_curr, device=s.device)
            time_embed = self.structure_time_embed(timestep_embedding(t, 64))
            vec = (context + time_embed + stage_embed).squeeze(1) # B, C
            z_embed = self.structure_embed(z)
            z_s = torch.cat([context, content, z_embed], dim=1) # B, 1+L+L, C
            if use_cfg:
                un_z_s = torch.cat([un_context, content, z_embed], dim=1)
                un_vec = (un_context + time_embed + stage_embed).squeeze(1)
                z_s = torch.cat((z_s, un_z_s), dim=0)
                vec = torch.cat((vec, un_vec), dim=0)
            z_s = self.forward_structure(z_s, vec, rope)
            v = self.structure_head(z_s[:, -self.num_tokens[-1]:], vec)
            if use_cfg:
                cond_v, uncond_v = torch.split(v, B, dim=0)
                v = uncond_v + cfg_scale * (cond_v - uncond_v)
            z = z + (t_prev - t_curr) * v
            z[:, :, :stage] = s[:, :, :stage]
        structure_pred = self.cluster(structure, z[:, :, stage], stage, use_gumbel_topk=use_gumbel_topk)
        return structure_pred

    def prepare_train_inputs_structure(self, structure, content, txt, batch_size):
        """Prepare inputs for the structure part of the model.
        Args:
            structure: Tensor of shape ((B S) L C) where B is the batch size, S is the number of stages, and L is the number of tokens, C is the number of embedded channels. Only last stage is used.
            content: Canvas of stage i, used to predict the structure of stage i+1. Remove first and last stage, which contains no tokens and 128 unique tokens, respectively. Their structures do not to be predicted.
            txt: class idx.
            batch_size: Batch size.
        Returns:
            epsilon: Noise.
            z: Noised image.
            vec: Context vector.
            s: Target.
        """
        s = rearrange(structure, "(b s) l c-> b s l c", b=batch_size)[:, -1:]
        epsilon = torch.randn_like(s)
        epsilon = epsilon.repeat(1, self.num_step-2, 1, 1)
        u = torch.randn((batch_size, self.num_step-2, 1, 1), device=epsilon.device)
        t = torch.nn.functional.sigmoid(u)
        s = s.repeat(1, self.num_step-2, 1, 1)
        s = s - 1 # [-1, 1]
        for i in range(1, self.num_step-2):
            epsilon[:, i, :, :i] = s[:, i, :, :i]
        z_t = t * epsilon + (1 - t) * s

        epsilon = rearrange(epsilon, "b s l c -> (b s) l c")
        z = self.structure_embed(z_t) # B, S-2, L, C
        t = rearrange(t, "b s 1 1-> (b s)")
        time_embed = self.structure_time_embed(timestep_embedding(t, 64))
        time_embed = rearrange(time_embed, "(b s) c -> b s c", b=batch_size)
        context = self.structure_class_embed(txt)
        vec = context + time_embed + self.structure_stage_embed.weight.unsqueeze(0).expand(batch_size, -1, -1)
        vec = rearrange(vec, "b s c -> (b s) c")
        context = context.unsqueeze(1).repeat(1, self.num_step-2, 1, 1)
        content = self.structure_content_cond_embed(content[:, 1:-1])
        z = torch.cat([context, content, z], dim=2)
        z = rearrange(z, "b s l c -> (b s) l c")
        return epsilon, z, vec, s

    def prepare_rope_inference(self, x, s, batch_size):
        """Prepare RoPE embedding for the inference. Add structure feature map to the RoPE embedding.
        Args:
            x: x_rope
            s: Structure feature map.
            batch_size: Batch size.
        Returns:
            x_rope_embed: RoPE embedding for the content part.
            s_rope_embed: RoPE embedding for the structure part.
        """
        x_rope = self.rope_ids.repeat(x.shape[0], 1, 1) # b l c
        x_rope[:, 1:, 1:self.num_step] = s
        x_rope_embed = self.rope_embedder(x_rope)
        s_rope = x_rope
        s_rope = torch.cat([s_rope, s_rope[:, 1:]], dim=1)
        s_rope[:, 1:1+self.num_tokens[-1], 0] = 2 # 2 is used to indicate the canvas condition for structure part.
        s_rope_embed = self.rope_embedder(s_rope)
        return x_rope_embed, s_rope_embed

    def prepare_rope_train(self, s, batch_size):
        """Prepare RoPE embedding for the training. Add structure feature map to the RoPE embedding.
        Remove the last two stages for the structure part.
        Since we first predict the structure then the content.
        In the last two stages, we predict the content of 128 and 256 with their structure.
        Hence they are removed for the structure part.
        Args:
            x: x_rope
            s: Structure feature map.
            batch_size: Batch size.
        Returns:
            x_rope_embed: RoPE embedding for the content part.
            s_rope_embed: RoPE embedding for the structure part.
        """
        x_rope = self.rope_ids.repeat(s.shape[0], 1, 1) # (b s) l c
        x_rope[:, 1:, 1:self.num_step] = s
        x_rope_embed = self.rope_embedder(x_rope)
        s_rope = rearrange(x_rope, "(b s) l c -> b s l c", b=batch_size)
        s_rope = s_rope[:, :-2]
        s_rope = rearrange(s_rope, "b s l c -> (b s) l c")
        s_rope = torch.cat([s_rope, s_rope[:, 1:]], dim=1)
        s_rope[:, 1:1+self.num_tokens[-1], 0] = 2
        s_rope_embed = self.rope_embedder(s_rope)
        return x_rope_embed, s_rope_embed

    @torch.no_grad()
    def get_current_canvas(self, logits, structure, stage):
        distance = torch.cdist(logits, self.embedding.weight)
        index = torch.argmin(distance, dim=-1)
        embed = self.embedding(index)
        embed = torch.gather(embed, 1, structure.unsqueeze(-1).expand(-1, -1, embed.shape[-1]))
        embed = rearrange(embed, "b (h w) c -> b c h w", h=int(self.num_tokens[-1]**0.5), w=int(self.num_tokens[-1]**0.5))
        canvas = self.quant_resi[stage/(self.num_step-1)](embed)
        canvas = rearrange(canvas, "b c h w -> b (h w) c")
        return canvas

    def bit_to_int(self, bits, stage):
        powers = 2 ** torch.arange(stage, -1, -1, device=bits.device)  # [C]
        return (bits * powers).sum(dim=-1)

    def generate(
        self,
        img,
        txt,
        stage = None,
        structure = None,
        structure_epsilon = None,
        temperature = 1.0,
        top_k = 0,
        top_p = 1.0,
        content_use_cfg = False,
        content_cfg_scale = 1.0,
        structure_use_cfg = False,
        structure_cfg_scale = 1.0,
        structure_sampling_step = 50,
        use_gumbel_topk = False,
    ):
        """Generate the next stage of the model.
        Args:
            img: Tensor of shape (B, L, C) where B is the batch size, L is the number of tokens, and C is the number of channels.
            txt: class idx.
            stage: Current stage of the model.
            structure: Structure of stage i.
            structure_epsilon: Start noise for the structure part.
        Returns:
            x0_hat: Tensor of shape (B, L, C) where B is the batch size, L is the number of tokens, and C is the number of channels.
            pred: Tensor of shape (B, L) where B is the batch size and L is the number of tokens.
            pred_inp: Tensor of shape (B, L, C) where B is the batch size, L is the number of tokens, and C is the number of channels.
            structure_pred: Tensor of shape (B, L) where B is the batch size and L is the number of tokens.
        """
        B = txt.shape[0]
        structure = structure.long()
        z_c, v_c, s = self.prepare_inference_inputs_content(img, structure, txt, B, stage)
        rope = self.prepare_rope_inference(z_c, s, B)
        rope_c, rope_s = rope

        if content_use_cfg:
            un_txt = self.num_classes * torch.ones_like(txt)
            un_zc, un_v_c, _ = self.prepare_inference_inputs_content(img, structure, un_txt, B, stage)
            z_c = torch.cat((z_c, un_zc), dim=0)
            v_c = torch.cat((v_c, un_v_c), dim=0)
            rope_c = torch.cat((rope_c, rope_c), dim=0)
            B = B * 2
            structure_used_in_generation = structure.repeat(2, 1)
        else:
            structure_used_in_generation = structure

        # content
        self.use_repa = False
        z_c, _ = self.forward_content(z_c, v_c, rope_c)

        z_c = z_c[:, 1:]

        v = self.x0_head(z_c, v_c)
        x0_hat = v / self.scale_factors[-1]
        previous_canvas = img
        if content_use_cfg:
            previous_canvas = previous_canvas.repeat(2, 1, 1)
        current_canvas = x0_hat - previous_canvas

        current_unique_logits = self.get_compressed_logits(current_canvas, structure_used_in_generation, stage)
        logits = self.cls_head(current_unique_logits, v_c)
        if content_use_cfg:
            cond_logits, uncond_logits = torch.split(logits, B // 2, dim=0)
            logits = uncond_logits + (cond_logits - uncond_logits) * content_cfg_scale
        if top_k == 1:
            sample_logits = False
        else:
            sample_logits = True
        pred = sample(logits, temperature=temperature, top_k=top_k, top_p=top_p, sample_logits=sample_logits)[0]
        pred = torch.gather(pred, 1, structure)
        pred_inp = self.nvg_next_input(img, pred, stage)

        # structure
        if stage < self.num_step - 1:
            structure_pred = self.structure_sampling(
                epsilon=structure_epsilon,
                content=pred_inp,
                txt=txt,
                s=s,
                structure=structure,
                stage=stage,
                rope=rope_s,
                use_cfg=structure_use_cfg,
                cfg_scale=structure_cfg_scale,
                sampling_step=structure_sampling_step,
                use_gumbel_topk=use_gumbel_topk,
            )
        else:
            structure_pred = None

        return x0_hat, pred, pred_inp, structure_pred

    @maybe_compile
    def forward_structure(self, z_s, v_s, rope_s):
        for i in range(self.depth):
            z_s = self.structure_blocks[i](z_s, vec=v_s, pe=rope_s)
        return z_s

    @maybe_compile
    def forward_content(self, z_c, v_c, rope_c):
        for i in range(self.depth):
            z_c = self.content_blocks[i](z_c, vec=v_c, pe=rope_c)
            if self.use_repa and (i+1) == self.repa_depth:
                repa_features = self.projector(z_c)

        if not self.use_repa:
            repa_features = None

        return z_c, repa_features

    def forward(
        self,
        img,
        txt,
        structure,
    ):
        """Forward pass of the model.
        Args:
            img: Tensor of shape (B, S, L, C) where B is the batch size, S is the number of stages, L is the number of tokens, and C is the number of channels.
            txt: class idx.
            structure: Tensor of shape (B, S, L). Structure of each stage
        Returns:
            x0_hat: Tensor of shape ((B, S), L, C) where B is the batch size, S is the number of stages, L is the number of tokens, and C is the number of channels.
            current_unique_logits: List of tensors of shape [(B, L, C) for each stage] where B is the batch size, L is the number of tokens, and C is the number of channels.
            structure_logits: Tensor of shape ((B, S-2), L, C) where B is the batch size, S is the number of stages, and L is the number of tokens, and C is the number of channels.
            repa_features: Tensor of shape ((B, S), L, C) where B is the batch size, S is the number of stages, L is the number of tokens, and C is the number of channels.
            s_gt: Structure feature map. ((B, S-2), L, C) where B is the batch size, S is the number of stages, and L is the number of tokens, and C is the number of channels.
        """
        B = txt.shape[0]
        structure = structure.long()
        z_c, v_c, s = self.prepare_train_inputs_content(img, structure, txt, B)

        rope = self.prepare_rope_train(s, B)
        e_s, z_s, v_s, s_gt = self.prepare_train_inputs_structure(s, img, txt, B)
        rope_c, rope_s = rope

        # structure
        z_s = self.forward_structure(z_s, v_s, rope_s)
        v = self.structure_head(z_s[:, 1+self.num_tokens[-1]:], v_s) # v: epsilon - structure
        structure_logits = e_s - v

        # content
        z_c, repa_features = self.forward_content(z_c, v_c, rope_c)
        z_c = z_c[:, 1:]
        if repa_features is not None:
            repa_features = repa_features[:, 1:]

        v = self.x0_head(z_c, v_c)
        x0_hat = v / self.scale_factors[-1]

        previous_canvas = rearrange(img, "b s l c -> (b s) l c")
        current_canvas = x0_hat - previous_canvas

        current_canvas = rearrange(current_canvas, "(B S) L C -> S B L C", B=B)
        vec_current_canvas = rearrange(v_c, "(B S) C -> S B C", B=B)
        structure = rearrange(structure, "B S L -> S B L", B=B).long()
        current_unique_logits = []
        for i in range(self.num_step):
            current_unique_logits_i = self.get_compressed_logits(current_canvas[i], structure[i], i)
            current_unique_logits_i = self.cls_head(current_unique_logits_i, vec_current_canvas[i])
            current_unique_logits.append(current_unique_logits_i)

        return x0_hat, current_unique_logits, structure_logits, repa_features, s_gt
