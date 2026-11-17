"""Fixed-ratio RGB-D Dinomaly2 with feature jitter and class decoupling.

The frozen DINOv2 encoder supplies the clean reconstruction targets. During
training, only spatial tokens entering the trainable bottleneck are perturbed
with the norm-scaled Gaussian feature jitter used by IB-IUMAD. A single
ten-way classifier reads the final shared decoder class token, so its
cross-entropy gradient updates the same bottleneck/decoder used for anomaly
reconstruction instead of an independent frozen-feature branch.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.uad import DinomalyRGBD


class DinomalyRGBDJitterClass(DinomalyRGBD):
    def __init__(
        self,
        *args,
        embed_dim: int,
        num_classes: int = 10,
        feature_jitter_scale: float = 20.0,
        feature_jitter_prob: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if feature_jitter_scale < 0:
            raise ValueError("feature_jitter_scale must be non-negative")
        if not 0.0 <= feature_jitter_prob <= 1.0:
            raise ValueError("feature_jitter_prob must be in [0, 1]")
        self.feature_jitter_scale = float(feature_jitter_scale)
        self.feature_jitter_prob = float(feature_jitter_prob)
        self.classifier_norm = nn.LayerNorm(embed_dim, elementwise_affine=False)
        self.classifier = nn.Linear(embed_dim, num_classes, bias=False)
        nn.init.trunc_normal_(self.classifier.weight, std=0.01)
        self.register_buffer("jitter_calls", torch.zeros((), dtype=torch.long), persistent=True)
        self.register_buffer("jitter_applied", torch.zeros((), dtype=torch.long), persistent=True)

    def _encode_rgbd(self, image: torch.Tensor):
        if image.ndim != 4 or image.shape[1] != 6:
            raise ValueError(f"RGB-D input must be Bx6xHxW, got {tuple(image.shape)}")
        batch = image.shape[0]
        paired = torch.cat((image[:, :3], image[:, 3:]), dim=0)
        with torch.no_grad():
            paired = self.encoder.prepare_tokens(paired)
        features = []
        for index, block in enumerate(self.encoder.blocks):
            if index > self.target_layers[-1]:
                break
            if index in self.encoder_require_grad_layer:
                paired = block(paired)
            else:
                with torch.no_grad():
                    paired = block(paired)
            if index in self.target_layers:
                features.append(paired)
        return [
            self.rgb_ratio * value[:batch] + (1.0 - self.rgb_ratio) * value[batch:]
            for value in features
        ]

    def add_feature_jitter(self, tokens: torch.Tensor, force: bool = False):
        """Apply the IB-IUMAD norm-scaled Gaussian jitter to spatial tokens.

        IB-IUMAD uses std = ||token||_2 / C * scale. Its reconstruction input
        contains spatial tokens only. Dinomaly2 additionally carries class and
        register tokens, so those tokens remain clean and only patch tokens are
        jittered.
        """
        self.jitter_calls.add_(1)
        should_apply = force or (
            self.training
            and self.feature_jitter_scale > 0.0
            and bool(torch.rand((), device=tokens.device) <= self.feature_jitter_prob)
        )
        if not should_apply:
            zero = tokens.new_zeros(())
            return tokens, zero, zero
        special = 1 + self.encoder.num_register_tokens
        patches = tokens[:, special:, :]
        channel_dim = patches.shape[-1]
        token_scale = patches.norm(dim=-1, keepdim=True) / float(channel_dim)
        noise = torch.randn_like(patches) * token_scale * self.feature_jitter_scale
        disturbed = torch.cat((tokens[:, :special, :], patches + noise), dim=1)
        self.jitter_applied.add_(1)
        feature_rms = patches.square().mean().sqrt().detach()
        jitter_rms = noise.square().mean().sqrt().detach()
        return disturbed, feature_rms, jitter_rms

    def forward(self, image: torch.Tensor, disable_jitter: bool = False):
        batch = image.shape[0]
        encoder_layers = self._encode_rgbd(image)
        side = int(math.sqrt(encoder_layers[0].shape[1] - 1 - self.encoder.num_register_tokens))

        clean_latent = self.fuse_feature(
            [encoder_layers[index] for index in self.fuse_layer_bottleneck]
        ).detach()
        if self.training and not disable_jitter:
            latent, feature_rms, jitter_rms = self.add_feature_jitter(clean_latent)
        else:
            latent = clean_latent
            feature_rms = clean_latent.new_zeros(())
            jitter_rms = clean_latent.new_zeros(())

        for block in self.bottleneck:
            latent = block(latent)
        mask = self.generate_mask(side, latent.device) if self.mask_neighbor_size > 0 else None

        decoder_layers = []
        decoded = latent
        for block in self.decoder:
            decoded = block(decoded, attn_mask=mask)
            decoder_layers.append(decoded)

        class_feature = self.classifier_norm(decoded[:, 0, :])
        class_logits = self.classifier(class_feature)
        decoder_layers = decoder_layers[::-1]

        encoder_groups = [
            self.fuse_feature([encoder_layers[index] for index in indices])
            for indices in self.fuse_layer_encoder
        ]
        decoder_groups = [
            self.fuse_feature([decoder_layers[index] for index in indices])
            for indices in self.fuse_layer_decoder
        ]
        special = 1 + self.encoder.num_register_tokens
        if self.add_class_token_weight != 0:
            encoder_groups = [
                value[:, :1, :] * self.add_class_token_weight + value[:, special:, :]
                for value in encoder_groups
            ]
        else:
            encoder_groups = [value[:, special:, :] for value in encoder_groups]
        decoder_groups = [value[:, special:, :] for value in decoder_groups]
        if self.norm_encoder_token:
            encoder_groups = [F.layer_norm(value, (value.shape[-1],), eps=1e-8) for value in encoder_groups]

        encoder_maps = [
            value.permute(0, 2, 1).reshape(batch, -1, side, side).contiguous()
            for value in encoder_groups
        ]
        decoder_maps = [
            value.permute(0, 2, 1).reshape(batch, -1, side, side).contiguous()
            for value in decoder_groups
        ]
        return {
            "en": encoder_maps,
            "de": decoder_maps,
            "class_logits": class_logits,
            "class_feature": class_feature,
            "feature_rms": feature_rms,
            "jitter_rms": jitter_rms,
        }

