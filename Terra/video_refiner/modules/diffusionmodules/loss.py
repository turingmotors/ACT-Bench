"""Code reference: https://github.com/OpenDriveLab/Vista/blob/main/vwm/modules/diffusionmodules/loss.py"""

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from video_refiner.modules.diffusionmodules.util import fourier_filter
from video_refiner.modules.encoders.modules import GeneralConditioner
from video_refiner.util import append_dims, instantiate_from_config
from .denoiser import Denoiser


class StandardDiffusionLoss(nn.Module):
    def __init__(
        self,
        sigma_sampler_config: dict,
        loss_weighting_config: dict,
        loss_type: str = "l2",
        use_additional_loss: bool = False,
        offset_noise_level: float = 0.0,
        additional_loss_weight: float = 0.0,
        num_frames: int = 25,
    ):
        super().__init__()
        assert loss_type in ["l2", "l1"]
        self.loss_type = loss_type
        self.use_additional_loss = use_additional_loss

        self.sigma_sampler = instantiate_from_config(sigma_sampler_config)
        self.loss_weighting = instantiate_from_config(loss_weighting_config)

        self.offset_noise_level = offset_noise_level
        self.additional_loss_weight = additional_loss_weight
        self.num_frames = num_frames

    def get_noised_input(
        self,
        sigma_bc: torch.Tensor,
        noise: torch.Tensor,
        input: torch.Tensor,
    ) -> torch.Tensor:
        noised_input = input + noise * sigma_bc
        return noised_input
    
    def forward(
        self,
        network: nn.Module,
        denoiser: Denoiser,
        conditioner: GeneralConditioner,
        input: torch.Tensor,
        batch: Dict
    ) -> torch.Tensor:
        cond = conditioner(batch)
        cond["vector"] = cond["vector"].bfloat16()
        return self._forward(network, denoiser, cond, input)

    def _forward(
        self,
        network: nn.Module,
        denoiser: Denoiser,
        cond: Dict,
        input: torch.Tensor,
    ):
        sigmas = self.sigma_sampler(input.shape[0]).to(input)
        cond_mask = torch.zeros_like(sigmas)
        noise = torch.randn_like(input)
        if self.offset_noise_level > 0.0:
            offset_shape = (input.shape[0], input.shape[1])
            rand_init = torch.randn(offset_shape, device=input.device, dtype=input.dtype)
            noise = noise + self.offset_noise_level * append_dims(rand_init, input.ndim)

        sigmas_bc = append_dims(sigmas, input.ndim)
        noised_input = self.get_noised_input(sigmas_bc, noise, input)

        model_output = denoiser(network, noised_input, sigmas, cond, cond_mask)
        w = append_dims(self.loss_weighting(sigmas), input.ndim)

        return self.get_loss(model_output, input, w)
    
    def get_loss(self, predict: torch.Tensor, target: torch.Tensor, w: torch.Tensor):
        if self.loss_type == "l2":
            if self.use_additional_loss:
                predict_seq = rearrange(predict, "(b t) ... -> b t ...", t=self.num_frames)
                target_seq = rearrange(target, "(b t) ... -> b t ...", t=self.num_frames)
                bs = target.shape[0] // self.num_frames
                aux_loss = ((target_seq[:, 1:] - target_seq[:, :-1]) - (predict_seq[:, 1:] - predict_seq[:, :-1])) ** 2
                tmp_h, tmp_w = aux_loss.shape[-2], aux_loss.shape[-1]
                aux_loss = rearrange(aux_loss, "b t c h w -> b (t h w) c", c=4)
                aux_w = F.normalize(aux_loss, p=2)
                aux_w = rearrange(aux_w, "b (t h w) c -> b t c h w", t=self.num_frames - 1, h=tmp_h, w=tmp_w)
                aux_w = 1 + torch.cat((torch.zeros(bs, 1, *aux_w.shape[2:]).to(aux_w), aux_w), dim=1)
                aux_w = rearrange(aux_w, "b t ... -> (b t) ...").reshape(target.shape[0], -1)
                predict_hf = fourier_filter(predict, scale=0.)
                target_hf = fourier_filter(target, scale=0.)
                hf_loss = torch.mean((w * (predict_hf - target_hf) ** 2).reshape(target.shape[0], -1), 1).mean()
                return torch.mean(
                    (w * (predict - target) ** 2).reshape(target.shape[0], -1) * aux_w.detach(), 1
                ).mean() + self.additional_loss_weight * hf_loss
            else:
                return torch.mean(
                    (w * (predict - target) ** 2).reshape(target.shape[0], -1), 1
                )
        elif self.loss_type == "l1":
            if self.use_additional_loss:
                predict_seq = rearrange(predict, "(b t) ... -> b t ...", t=self.num_frames)
                target_seq = rearrange(target, "(b t) ... -> b t ...", t=self.num_frames)
                bs = target.shape[0] // self.num_frames
                aux_loss = ((target_seq[:, 1:] - target_seq[:, :-1]) - (predict_seq[:, 1:] - predict_seq[:, :-1])).abs()
                tmp_h, tmp_w = aux_loss.shape[-2], aux_loss.shape[-1]
                aux_loss = rearrange(aux_loss, "b t c h w -> b (t h w) c", c=4)
                aux_w = F.normalize(aux_loss, p=1)
                aux_w = rearrange(aux_w, "b (t h w) c -> b t c h w", t=self.num_frames - 1, h=tmp_h, w=tmp_w)
                aux_w = 1 + torch.cat((torch.zeros(bs, 1, *aux_w.shape[2:]).to(aux_w), aux_w), dim=1)
                aux_w = rearrange(aux_w, "b t ... -> (b t) ...").reshape(target.shape[0], -1)
                predict_hf = fourier_filter(predict, scale=0.)
                target_hf = fourier_filter(target, scale=0.)
                hf_loss = torch.mean((w * (predict_hf - target_hf).abs()).reshape(target.shape[0], -1), 1).mean()
                return torch.mean(
                    (w * (predict - target).abs()).reshape(target.shape[0], -1) * aux_w.detach(), 1
                ).mean() + self.additional_loss_weight * hf_loss
            else:
                return torch.mean(
                    (w * (predict - target).abs()).reshape(target.shape[0], -1), 1
                )
        else:
            raise NotImplementedError(f"Unknown loss type {self.loss_type}")
