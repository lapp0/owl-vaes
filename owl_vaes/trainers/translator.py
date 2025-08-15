import os
import time
from pathlib import Path
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import wandb

# TODO: EMA (details) — reintroduce when we want moving-avg weights

from ..utils.logging import to_wandb_gif, LogHelper
from ..utils import Timer
from ..data import get_loader
from ..models import get_model_cls
from ..schedulers import get_scheduler_cls
from ..muon import init_muon
from ..configs import Config

from .base import BaseTrainer



################################################################################################

from torch.nn.utils.parametrizations import weight_norm
import einops as eo

class Upsample(nn.Module):
    """
    Bilinear upsample + project layer
    """
    def __init__(self, ch_in, ch_out):
        super().__init__()

        self.proj = nn.Sequential() if ch_in == ch_out else weight_norm(nn.Conv2d(ch_in, ch_out, 1, 1, 0, bias=False))

    def forward(self, x):
        x = self.proj(x)
        x = F.interpolate(x, scale_factor = 2,  mode = 'bicubic')
        return x


class LearnableUpsample(nn.Module):
    """
    Learnable upsampling using transposed convolution
    """
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.proj = nn.Sequential() if ch_in == ch_out else weight_norm(nn.Conv2d(ch_in, ch_out, 1, 1, 0, bias=False))
        self.upsample = nn.ConvTranspose2d(ch_out, ch_out, kernel_size=4, stride=2, padding=1, bias=False)

    def forward(self, x):
        x = self.proj(x)
        x = self.upsample(x)
        return x


class UpBlock(nn.Module):
    """
    General upsampling stage block
    """
    def __init__(self, ch_in, ch_out, num_res, total_blocks, learnable_upsample=True):
        super().__init__()

        self.up = LearnableUpsample(ch_in, ch_out) if learnable_upsample else Upsample(ch_in, ch_out)
        blocks = []
        num_total = num_res * total_blocks
        for _ in range(num_res):
            blocks.append(ResBlock(ch_out, num_total))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        x = self.up(x)
        for block in self.blocks:
            x = block(x)
        return x


class ResBlock(nn.Module):
    """
    Basic ResNet block from R3GAN paper using

    :param ch: Channel count to use for the block
    :param total_res_blocks: How many res blocks are there in the entire model?
    """
    def __init__(self, ch, total_res_blocks, act_fn="silu"):
        super().__init__()

        grp_size = 16
        n_grps = (2*ch) // grp_size

        self.conv1 = weight_norm(nn.Conv2d(ch, 2*ch, 1, 1, 0))
        #self.norm1 = RMSNorm2d(2*ch)
        #self.norm1 = GroupNorm(2*ch, n_grps)

        self.conv2 = weight_norm(nn.Conv2d(2*ch, 2*ch, 3, 1, 1, groups = n_grps))
        #self.norm2 = RMSNorm2d(2*ch)
        #self.norm2 = GroupNorm(2*ch, n_grps)

        self.conv3 = weight_norm(nn.Conv2d(2*ch, ch, 1, 1, 0, bias=False))

        if act_fn == "leaky_relu":
            self.act1 = nn.LeakyReLU(inplace=True)
            self.act2 = nn.LeakyReLU(inplace=True)
        elif act_fn == "gelu":
            self.act1 = nn.GELU()
            self.act2 = nn.GELU()
        elif act_fn == "silu":
            self.act1 = nn.SiLU(inplace=True)
            self.act2 = nn.SiLU(inplace=True)
        else:
            raise ValueError(f"Invalid activation function: {act_fn}")

        # Fix up init
        scaling_factor = total_res_blocks ** -.25

        nn.init.kaiming_uniform_(self.conv1.weight)
        nn.init.zeros_(self.conv1.bias)
        self.conv1.weight.data *= scaling_factor

        nn.init.kaiming_uniform_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)
        self.conv2.weight.data *= scaling_factor

        nn.init.zeros_(self.conv3.weight)

    def forward(self, x):
        res = x.clone()

        def _inner(x):
            x = self.conv1(x)
            #x = self.norm1(x)
            x = self.act1(x)
            x = self.conv2(x)
            #x = self.norm2(x)
            x = self.act2(x)
            x = self.conv3(x)
            return x

        if False and self.training:
            x = checkpoint(_inner, x)
        else:
            x = _inner(x)

        return x + res

class SameBlock(nn.Module):
    """
    General block with no up/down
    """
    def __init__(self, ch_in, ch_out, num_res, total_blocks):
        super().__init__()

        blocks = []
        num_total = num_res * total_blocks
        for _ in range(num_res):
            blocks.append(ResBlock(ch_in, num_total))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class LearnableTemporalAggregation(nn.Module):
    """
    Learnable temporal aggregation module that replaces simple averaging.
    Uses attention-based weighted combination of temporal frames.
    """
    def __init__(self, channels, group_size=4):
        super().__init__()
        self.group_size = group_size
        self.channels = channels

        # Attention mechanism for temporal aggregation
        self.temporal_attention = nn.Conv1d(channels, channels, kernel_size=group_size, stride=group_size)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (b, n, c, h, w) where n-1 is divisible by group_size
        Returns:
            Aggregated tensor of shape (b, 1 + (n-1)//group_size, c, h, w)
        """
        b, n, c, h, w = x.shape
        y = eo.rearrange(x, 'b n c h w -> (b h w) c n')
        y = self.temporal_attention(y)
        y = eo.rearrange(y, '(b h w) c n1 -> b n1 c h w', b=b,c=c,h=h,w=w,n1=n//4)
        return y

class LatentTranslator(nn.Module):
    """
    Translator that converts latents from 101x128x4x4 to 26x16x64x64.
    Uses the same residual blocks and upblocks as the DCAE architecture.
    More aggressive channel reduction with learnable temporal aggregation and normalization.
    """
    def __init__(self, cfg, *args, **kwargs):
        super().__init__()
        self.input_channels = cfg.input_channels
        self.output_channels = cfg.output_channels
        self.input_size = cfg.input_size
        self.output_size = cfg.output_size
        self.d_model = cfg.d_model
        self.group_size = cfg.group_size
        self.ftemporal = cfg.ftemporal
        self.use_weight_norm = False
        # Default same blocks configuration if not provided
        self.same_blocks_per_stage = [cfg.n_same_per_stage]*5

        self.total_blocks = sum(self.same_blocks_per_stage)
        # Upsample from 4x4 to 64x64 (16x upsampling)
        self.upsample_factor = cfg.output_size // cfg.input_size  # 16

        # Initial channel reduction with normalization
        if self.use_weight_norm:
            self.conv_in = weight_norm(nn.Conv2d(cfg.input_channels, cfg.d_model, kernel_size=3, stride=1, padding=1))
        else:
            self.conv_in = nn.Conv2d(cfg.input_channels, cfg.d_model, kernel_size=3, stride=1, padding=1)

        # Progressive upsampling using UpBlock pattern with normalization, all in self.body
        self.body = nn.ModuleList([
            nn.GroupNorm(cfg.group_size, cfg.d_model),
            SameBlock(cfg.d_model, cfg.d_model, num_res=self.same_blocks_per_stage[0], total_blocks=self.total_blocks),
            nn.GroupNorm(cfg.group_size, cfg.d_model),
            UpBlock(cfg.d_model, cfg.d_model, num_res=self.same_blocks_per_stage[1], total_blocks=self.total_blocks),
            nn.GroupNorm(cfg.group_size, cfg.d_model),
            UpBlock(cfg.d_model, cfg.d_model, num_res=self.same_blocks_per_stage[2], total_blocks=self.total_blocks),
            nn.GroupNorm(cfg.group_size, cfg.d_model),
            UpBlock(cfg.d_model, cfg.d_model, num_res=self.same_blocks_per_stage[3], total_blocks=self.total_blocks),
            nn.GroupNorm(cfg.group_size, cfg.d_model),
            UpBlock(cfg.d_model, cfg.d_model, num_res=self.same_blocks_per_stage[4], total_blocks=self.total_blocks)
        ])

        # Final refinement with normalization
        if self.use_weight_norm:
            self.final = weight_norm(nn.Conv2d(cfg.d_model, cfg.output_channels, kernel_size=3, stride=1, padding=1))
        else:
            self.final = nn.Conv2d(cfg.d_model, cfg.output_channels, kernel_size=3, stride=1, padding=1)

        # Learnable temporal aggregation
        self.temporal_aggregator = LearnableTemporalAggregation(cfg.output_channels, group_size=cfg.ftemporal)


    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch, n, 128, 4, 4)
        Returns:
            Output tensor of shape (batch, n, 16, 64, 64)
        """
        # Initial channel reduction
        assert (x.shape[1])%4 == 0, f"{x.shape} sequence length must be divisible by 4"
        b, n, c, h, w = x.shape
        x_flat = eo.rearrange(x, 'b n c h w -> (b n) c h w')

        x = self.conv_in(x_flat)
        for i, block in enumerate(self.body):
            x = block(x)
        x = self.final(x)

        x = eo.rearrange(x, '(b n) c h w -> b n c h w', b=b,n=n)
        x = self.temporal_aggregator(x)
        return x

    def translate_batch(self, x):
        """
        Translate a batch of latents from 101x128x4x4 to 26x16x64x64.
        Uses learnable temporal aggregation instead of simple averaging.

        Args:
            x: Input tensor of shape (b, 101, 128, 4, 4)
        Returns:
            Output tensor of shape (b, 26, 16, 64, 64)
        """
        assert (x.shape[1])%4 == 0, f"{x.shape} temporal dim must be divisible by 4"
        return self.forward(x)


################################################################################################



class Preprocessor:
    def __init__(self, owl_landscape_size=(360, 640), wan_square_size=(512, 512), drop_first_frame: bool = False):
        self.owl_landscape_size = tuple(owl_landscape_size)
        self.wan_square_size = tuple(wan_square_size)
        self.drop_first_frame = drop_first_frame

    def _resize(self, x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        b, t, c, h, w = x.shape
        y = F.interpolate(x.reshape(b * t, c, h, w), size, mode="bilinear", align_corners=False)
        return y.reshape(b, t, c, size[0], size[1])

    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [B,T,3,H,W] → (x_wan, x_owl) both in [-1,1], reference-parity policy
        _, _, _, h, w = x.shape
        sq = self.wan_square_size
        ls = self.owl_landscape_size

        # landscape / square / portrait
        if w > h:  # strictly landscape (fix: don't treat square as landscape)
            x_wan = self._resize(x, sq)
            x_owl = x if (h, w) == ls else self._resize(x, ls)
        elif (h, w) == sq:
            x_wan = x
            x_owl = x
        else:
            x_wan = self._resize(x, sq)
            x_owl = self._resize(x, sq)

        x_wan = x_wan.clamp(0, 255)
        x_owl = x_owl.clamp(0, 255)

        if self.drop_first_frame and x_wan.shape[1] > 1:
            x_wan = x_wan[:, 1:]
            x_owl = x_owl[:, 1:]

        # per-sample min–max to [-1,1], separately for WAN and OWL streams
        def norm(z: torch.Tensor) -> torch.Tensor:
            z = z.float()
            zmin = z.amin(dim=(1, 2, 3, 4), keepdim=True)
            zmax = z.amax(dim=(1, 2, 3, 4), keepdim=True)
            z = (z - zmin) / (zmax - zmin).clamp_min(1e-6)
            return z.mul(2).sub(1).clamp(-1, 1).to(torch.bfloat16)

        return norm(x_wan), norm(x_owl)


class OnlineLatentBridge(nn.Module):
    """Bidirectional OWL DCAE <-> WAN translator"""

    def __init__(
        self,
        owl_ae: nn.Module,
        wan_ae: nn.Module,
        owl_to_wan_translator: nn.Module,
        *,
        owl_landscape_size=(360, 640),
        wan_square_size=(512, 512),
    ) -> None:
        super().__init__()
        self.owl_ae = owl_ae.eval()
        self.wan_ae = wan_ae.eval()
        for p in self.owl_ae.parameters():
            p.requires_grad_(False)
        for p in self.wan_ae.parameters():
            p.requires_grad_(False)

        self.owl_to_wan = owl_to_wan_translator
        self.group = self.owl_to_wan.ftemporal
        assert self.group > 0

        self.pp = Preprocessor(owl_landscape_size, wan_square_size)

    @torch.no_grad()
    def _encode_owl(self, x_owl: torch.Tensor) -> torch.Tensor:
        b, t, _, h, w = x_owl.shape
        z = self.owl_ae.encoder(x_owl.reshape(b * t, 3, h, w))
        return z.view(b, t, *z.shape[1:])  # [B,T,Co,4,4]

    @torch.no_grad()
    def _encode_wan(self, x_wan: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = x_wan.shape
        # WAN expects first frame uncompressed, then /4 compression
        assert (t - 1) % 4 == 0, f"WAN temporal contract violated: T={t} ⇒ (T-1)%4 must be 0"
        z = self.wan_ae.encode(x_wan.movedim(2, 1).contiguous()).latent_dist.mode()
        z = z.movedim(2, 1).contiguous()
        expected_t = 1 + (t - 1) // 4
        assert z.shape[1] == expected_t, f"WAN encode produced T={z.shape[1]}, expected {expected_t}"
        return z  # [B,Tw,Cw,64,64]

    @torch.no_grad()
    def _decode_wan(self, lat: torch.Tensor) -> torch.Tensor:
        y = self.wan_ae.decode(lat.movedim(2, 1).contiguous())
        return y.movedim(2, 1).contiguous()

    @torch.no_grad()
    def _decode_owl(self, lat: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = lat.shape
        y = self.owl_ae.decode(lat.reshape(b * t, c, h, w))
        return y.view(b, t, 3, *y.shape[-2:])

    def train(self, mode: bool = True) -> "OnlineLatentBridge":
        self.owl_to_wan.train(mode)
        self.owl_ae.eval()
        self.wan_ae.eval()
        return self

    def forward(
        self,
        x_rgb: torch.Tensor,
        *,
        loss_reduction: str = "mean",
        loss_only: bool = True,
    ):
        assert x_rgb.ndim == 5 and x_rgb.shape[2] == 3, str(x_rgb.shape)

        # preprocess + encode
        x_wan, x_owl = self.pp(x_rgb)
        tgt_owl = self._encode_owl(x_owl)  # [B,T,Co,4,4]
        tgt_wan = self._encode_wan(x_wan)  # [B,Tw,Cw,64,64]

        # match reference translator training:
        # use WAN targets without the first (uncompressed) step
        tgt_wan = tgt_wan[:, 1:]  # [B,Tw,Cw,64,64] where Tw = (Td)//ftemporal

        # align DCAE time to the translator's downsampling (Td' = ftemporal * Tw)
        req_td = self.group * tgt_wan.shape[1]
        assert tgt_owl.shape[1] >= req_td, f"DCAE temporal too short: have {tgt_owl.shape[1]}, need {req_td}"
        if tgt_owl.shape[1] != req_td:
            tgt_owl = tgt_owl[:, :req_td]

        # forward translators
        assert tgt_owl.shape[1] % self.group == 0, f"Temporal len {tgt_owl.shape[1]} must be divisible {self.group}"
        dcae_in, tgt_wan = tgt_owl, tgt_wan
        pred_wan = self.owl_to_wan(dcae_in)
        assert pred_wan.shape[1] == tgt_wan.shape[1], "Temporal reduction mismatch for owl->wan"

        # losses
        reduce = torch.mean if loss_reduction == "mean" else torch.sum
        loss = reduce((pred_wan - tgt_wan) ** 2)

        if loss_only:
            return loss

        to_rgb = lambda x: x.float().add(1).mul(0.5).clamp(0, 1)
        wan_rgb = to_rgb(self._decode_wan(pred_wan))

        return {
            "loss": loss,
            "pred_wan": pred_wan,
            "tgt_wan": tgt_wan,
            "tgt_owl": tgt_owl,
            "wan_rgb": wan_rgb,
        }


class OnlineLatentTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.total_step_counter = 0
        self.opt = None
        self.scheduler = None

    # ---- helpers like reference ----
    @staticmethod
    def get_raw_model(model):
        return getattr(model, "module", model)

    def save(self):
        if self.rank != 0:
            return
        save_dict = {
            "model": self.get_raw_model(self.model).state_dict(),
            "ema": {},
            "opt": self.opt.state_dict() if self.opt is not None else {},
            "steps": self.total_step_counter,
        }
        #if self.scheduler is not None:
        #    save_dict["scheduler"] = self.scheduler.state_dict()
        super().save(save_dict)

    def load(self):
        """Build runtime objects and optionally restore a checkpoint."""
        # Hardcode AE - TODO  fix
        from diffusers import AutoencoderKLWan
        wan_ae = AutoencoderKLWan.from_pretrained(
            self.model_cfg.wan_repo,
            subfolder="vae",
            torch_dtype=torch.bfloat16
        )

        #### HACK
        from owl_vaes.configs import ResNetConfig
        from owl_vaes.models.dcae import DCAE
        cfg = ResNetConfig(
            sample_size=[360,640],
            channels=3,
            latent_size=4,
            latent_channels=128,
            noise_decoder_inputs=0.0,
            ch_0=256,
            ch_max=2048,
            encoder_blocks_per_stage = [4, 4, 4, 4, 4, 4, 4],
            decoder_blocks_per_stage = [4, 4, 4, 4, 4, 4, 4]
        )
        cfg.use_middle_block = False
        owl_ae = DCAE(cfg)
        ####
        # cfg = Config.from_yaml(self.model_cfg.vae_cfg_path).model
        # cfg.use_middle_block = False  # TODO: hack
        # owl_ae = get_model_cls(cfg.model_id)(cfg)

        owl_ae.load_state_dict(torch.load(self.model_cfg.vae_ckpt_path, map_location='cpu', weights_only=False))

        owl_ae.eval()

        for p in owl_ae.parameters():
            p.requires_grad_(False)
        for p in wan_ae.parameters():
            p.requires_grad_(False)

        owl_to_wan = LatentTranslator(self.model_cfg).to(torch.bfloat16)

        self.model = OnlineLatentBridge(
            owl_ae=owl_ae,
            wan_ae=wan_ae,
            owl_to_wan_translator=owl_to_wan,
            owl_landscape_size=self.model_cfg.owl_landscape_size,
            wan_square_size=self.model_cfg.wan_square_size,
        ).cuda()

        # DDP + compile (like reference)
        if self.world_size > 1:
            self.model = DDP(self.model, device_ids=[self.local_rank], find_unused_parameters=True)
        self.model = torch.compile(self.model)

        # ---- Optimiser (AdamW) ----
        params = [p for p in self.model.parameters() if p.requires_grad]
        assert self.train_cfg.opt == "AdamW", f"{self.train_cfg.opt} not implemented"
        self.opt = torch.optim.AdamW(params, **self.train_cfg.opt_kwargs)

        # TODO: continue checkpoint
        """
        ckpt = getattr(self.train_cfg, "resume_ckpt", None)
        if ckpt:
            state = super().load(ckpt)
            # Strip potential prefixes: module./_orig_mod./ema_model.
            import re
            pat = r'^(?:(?:_orig_mod\.|module\.)+)?([^.]+\.)?(?:(?:_orig_mod\.|module\.)+)?'
            state["model"] = {re.sub(pat, r'\1', k): v for k, v in state["model"].items()}
            self.get_raw_model(self.model).load_state_dict(state["model"], strict=True)
            self.total_step_counter = state.get("steps", 0)
            if "opt" in state and self.opt is not None:
                self.opt.load_state_dict(state["opt"])
            #if "scheduler" in state and self.scheduler is not None:
            #    self.scheduler.load_state_dict(state["scheduler"])
            del state
        """

    def train(self):
        torch.cuda.set_device(self.local_rank)
        print(f"Device used: rank={self.rank}")

        # grad-accum like reference
        accum_steps = self.train_cfg.target_batch_size // self.train_cfg.batch_size // self.world_size
        accum_steps = max(1, accum_steps)

        ctx = torch.amp.autocast('cuda', torch.bfloat16)

        self.load()

        # data
        loader = get_loader(self.train_cfg.data_id, self.train_cfg.batch_size, **self.train_cfg.data_kwargs)

        n_samples = (getattr(self.train_cfg, "n_samples", 4) + self.world_size - 1) // self.world_size
        sample_loader = get_loader(self.train_cfg.sample_data_id, n_samples, **self.train_cfg.sample_data_kwargs)
        sample_loader = iter(sample_loader)

        if self.rank == 0:
            wandb.watch(self.get_module(), log='all')

        local_step = 0
        timer = Timer()
        timer.reset()
        metrics = LogHelper()

        for batch in loader:
            # batch expected: [B,T,3,H,W] or (x, ...)
            print(len(batch))
            x_rgb = batch[0]
            #x_rgb = batch[0] if isinstance(batch, (tuple, list)) else batch
            x_rgb = x_rgb.cuda(non_blocking=True)

            with ctx:
                loss = self.model(x_rgb, loss_only=True)
                loss = loss / accum_steps
                loss.backward()

            metrics.log("loss", loss)

            local_step += 1
            if local_step % accum_steps == 0:
                # TODO: do we need gradient clipping? Probably not
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.train_cfg.grad_clip)
                self.opt.step()
                self.opt.zero_grad(set_to_none=True)

                self.total_step_counter += 1

                if self.logging_cfg is not None and self.rank == 0:
                    log = metrics.pop()
                    log["step"] = self.total_step_counter
                    log["time"] = timer.hit(); timer.reset()
                    wandb.log(log)

                # periodic sampling via eval_step (EMA)
                if (self.total_step_counter % self.train_cfg.sample_interval == 0) and sample_loader is not None:
                    with ctx, torch.no_grad():
                        eval_wandb = self.eval_step(sample_loader)
                        if self.rank == 0 and eval_wandb is not None:
                            wandb.log(eval_wandb)

                # checkpoint
                if self.total_step_counter % self.train_cfg.save_interval == 0:
                    self.save()

                self.barrier()

            if self.total_step_counter >= self.train_cfg.max_steps:
                if self.rank == 0:
                    self.save()
                break

        self.barrier()

    def _gather_concat_cpu(self, t: torch.Tensor, dim: int = 0):
        if self.world_size == 1:
            return t.detach().cpu()
        if self.rank == 0:
            parts = [t.detach().cpu()]
            scratch = torch.empty_like(t)
            for src in range(1, self.world_size):
                dist.recv(scratch, src=src)
                parts.append(scratch.detach().cpu())
            return torch.cat(parts, dim=dim)
        dist.send(t, dst=0)

    def eval_step(self, sample_loader):
        batch = next(sample_loader)

        x_rgb = batch[0]
        #x_rgb = batch[0] if isinstance(batch, (tuple, list)) else batch
        x_rgb = x_rgb.cuda(non_blocking=True)

        model = self.get_module()
        with torch.no_grad():
            out = model(x_rgb, loss_only=False)

        wan_rgb = out.get("wan_rgb")

        eval_dict = {}
        if wan_rgb is not None:
            wan_vid = self._gather_concat_cpu(wan_rgb).clamp(0, 1)        # [B,T,3,H,W]
            eval_dict["eval/wan_gif"] = to_wandb_gif(wan_vid * 2 - 1, max_samples=4)

        dist.barrier()
        return eval_dict
