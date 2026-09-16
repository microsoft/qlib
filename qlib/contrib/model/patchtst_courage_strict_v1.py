"""Shared PatchTST model frozen for the Courage-strict mainboard route."""

from __future__ import annotations

import torch
from torch import nn

HORIZONS_V1 = (5, 15, 30, 60, 120, 240, 480)
DYNAMIC_CHANNELS_V1 = 12
SLOW_CHANNELS_V1 = 5
LOOKBACK_V1 = 1200
PATCH_LENGTH_V1 = 30
PATCH_STRIDE_V1 = 15
PATCH_COUNT_V1 = 79
D_MODEL_V1 = 128
PATCH_INPUT_WIDTH_V1 = 121


class CourageStrictModelError(RuntimeError):
    """Raised when a model-facing tensor violates the C1 contract."""


class _QueryPool(nn.Module):
    def __init__(self, *, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(D_MODEL_V1)
        self.attention = nn.MultiheadAttention(
            D_MODEL_V1, 4, dropout=dropout, batch_first=True
        )
        self.query = nn.Parameter(torch.empty(1, 1, D_MODEL_V1))
        self.dropout = nn.Dropout(dropout)
        self.output_norm = nn.LayerNorm(D_MODEL_V1)

    def forward(
        self, tokens: torch.Tensor, *, key_padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        query = self.query.expand(tokens.shape[0], -1, -1)
        normalized = self.norm(tokens)
        pooled, _ = self.attention(
            query,
            normalized,
            normalized,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        return self.output_norm(query + self.dropout(pooled)).squeeze(1)


class PatchTSTCourageStrictV1(nn.Module):
    """Twelve minute channels + five T-1 values + PIT industry embedding."""

    horizons_minutes = HORIZONS_V1

    def __init__(
        self, *, industry_vocab_size: int, industry_embedding_dim: int = 16
    ) -> None:
        super().__init__()
        if industry_vocab_size < 2 or industry_embedding_dim <= 0:
            raise CourageStrictModelError("invalid industry embedding geometry")
        dropout = 0.1
        self.industry_vocab_size = int(industry_vocab_size)
        self.industry_embedding_dim = int(industry_embedding_dim)
        self.patch_projection = nn.Linear(PATCH_INPUT_WIDTH_V1, D_MODEL_V1)
        self.patch_position = nn.Parameter(
            torch.empty(1, 1, PATCH_COUNT_V1, D_MODEL_V1)
        )
        self.minute_embedding = nn.Embedding(240, D_MODEL_V1)
        self.session_embedding = nn.Embedding(2, D_MODEL_V1)
        layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL_V1,
            nhead=4,
            dim_feedforward=256,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            layer,
            num_layers=3,
            norm=nn.LayerNorm(D_MODEL_V1),
            enable_nested_tensor=False,
        )
        self.temporal_pool = _QueryPool(dropout=dropout)
        self.channel_identity = nn.Embedding(DYNAMIC_CHANNELS_V1, D_MODEL_V1)
        self.channel_norm = nn.LayerNorm(D_MODEL_V1)
        self.channel_attention = nn.MultiheadAttention(
            D_MODEL_V1, 4, dropout=dropout, batch_first=True
        )
        self.channel_pool = _QueryPool(dropout=dropout)
        self.slow_encoder = nn.Sequential(
            nn.Linear(2 * SLOW_CHANNELS_V1, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.LayerNorm(32),
        )
        self.industry_embedding = nn.Embedding(
            self.industry_vocab_size, self.industry_embedding_dim
        )
        self.fusion = nn.Sequential(
            nn.LayerNorm(D_MODEL_V1 + 32 + self.industry_embedding_dim),
            nn.Linear(D_MODEL_V1 + 32 + self.industry_embedding_dim, D_MODEL_V1),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.heads = nn.ModuleDict(
            {
                str(horizon): nn.Sequential(
                    nn.Linear(D_MODEL_V1, 64),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(64, 1),
                )
                for horizon in HORIZONS_V1
            }
        )
        self._initialize()

    @property
    def parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    def _initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.MultiheadAttention):
                if module.in_proj_weight is not None:
                    nn.init.xavier_uniform_(module.in_proj_weight)
                if module.in_proj_bias is not None:
                    nn.init.zeros_(module.in_proj_bias)
        nn.init.normal_(self.patch_position, mean=0.0, std=0.02)
        nn.init.normal_(self.temporal_pool.query, mean=0.0, std=0.02)
        nn.init.normal_(self.channel_pool.query, mean=0.0, std=0.02)

    @staticmethod
    def _validate(
        values: torch.Tensor,
        available: torch.Tensor,
        missing: torch.Tensor,
        padding: torch.Tensor,
        minute_index: torch.Tensor,
        session_index: torch.Tensor,
        slow_values: torch.Tensor,
        slow_available: torch.Tensor,
        industry_id: torch.Tensor,
        industry_vocab_size: int,
    ) -> None:
        if values.ndim != 3 or values.shape[1:] != (LOOKBACK_V1, DYNAMIC_CHANNELS_V1):
            raise CourageStrictModelError("dynamic shape drift")
        batch = values.shape[0]
        if available.shape != values.shape or missing.shape != values.shape:
            raise CourageStrictModelError("dynamic mask shape drift")
        if available.dtype != torch.bool or missing.dtype != torch.bool:
            raise CourageStrictModelError("dynamic masks must be bool")
        if padding.shape != (batch, LOOKBACK_V1) or padding.dtype != torch.bool:
            raise CourageStrictModelError("padding shape drift")
        if minute_index.shape != padding.shape or session_index.shape != padding.shape:
            raise CourageStrictModelError("time identity shape drift")
        if minute_index.dtype != torch.long or session_index.dtype != torch.long:
            raise CourageStrictModelError("time identities must be long")
        if slow_values.shape != (batch, SLOW_CHANNELS_V1):
            raise CourageStrictModelError("slow value shape drift")
        if (
            slow_available.shape != slow_values.shape
            or slow_available.dtype != torch.bool
        ):
            raise CourageStrictModelError("slow availability shape drift")
        if industry_id.shape != (batch,) or industry_id.dtype != torch.long:
            raise CourageStrictModelError("industry identity shape drift")
        if industry_id.lt(0).any() or industry_id.ge(industry_vocab_size).any():
            raise CourageStrictModelError("industry identity outside vocabulary")
        if (~padding).sum(dim=1).lt(1200).any():
            raise CourageStrictModelError(
                "C4 samples must have a full 1200-minute window"
            )
        usable = (~padding.unsqueeze(-1)) & available & ~missing
        if (
            not torch.isfinite(values).all()
            or values.masked_select(~usable).ne(0).any()
        ):
            raise CourageStrictModelError("dynamic value/mask invariant failed")
        if (
            not torch.isfinite(slow_values).all()
            or slow_values.masked_select(~slow_available).ne(0).any()
        ):
            raise CourageStrictModelError("slow value/mask invariant failed")

    def forward(
        self,
        values: torch.Tensor,
        available: torch.Tensor,
        missing: torch.Tensor,
        padding: torch.Tensor,
        minute_index: torch.Tensor,
        session_index: torch.Tensor,
        slow_values: torch.Tensor,
        slow_available: torch.Tensor,
        industry_id: torch.Tensor,
    ) -> torch.Tensor:
        self._validate(
            values,
            available,
            missing,
            padding,
            minute_index,
            session_index,
            slow_values,
            slow_available,
            industry_id,
            self.industry_vocab_size,
        )
        batch = values.shape[0]
        real = ~padding

        def patches(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.unfold(1, PATCH_LENGTH_V1, PATCH_STRIDE_V1).permute(
                0, 2, 1, 3
            )

        real_patches = real.unfold(1, PATCH_LENGTH_V1, PATCH_STRIDE_V1)
        fraction = real_patches.to(values.dtype).mean(dim=-1)
        patch_input = torch.cat(
            (
                patches(values),
                patches(available).to(values.dtype),
                patches(missing).to(values.dtype),
                real_patches[:, None]
                .expand(-1, DYNAMIC_CHANNELS_V1, -1, -1)
                .to(values.dtype),
                fraction[:, None, :, None].expand(-1, DYNAMIC_CHANNELS_V1, -1, -1),
            ),
            dim=-1,
        )
        if patch_input.shape[2:] != (PATCH_COUNT_V1, PATCH_INPUT_WIDTH_V1):
            raise CourageStrictModelError("patch geometry drift")
        time = self.minute_embedding(minute_index) + self.session_embedding(
            session_index
        )
        time_patches = time.unfold(1, PATCH_LENGTH_V1, PATCH_STRIDE_V1).permute(
            0, 1, 3, 2
        )
        count = real_patches.unsqueeze(-1).to(time.dtype).sum(dim=2)
        time_mean = (time_patches * real_patches.unsqueeze(-1).to(time.dtype)).sum(
            dim=2
        ) / count.clamp_min(1)
        tokens = self.patch_projection(patch_input)
        tokens = tokens + self.patch_position + time_mean[:, None]
        tokens = tokens.reshape(batch * DYNAMIC_CHANNELS_V1, PATCH_COUNT_V1, D_MODEL_V1)
        patch_padding = fraction.eq(0)[:, None].expand(-1, DYNAMIC_CHANNELS_V1, -1)
        patch_padding = patch_padding.reshape(
            batch * DYNAMIC_CHANNELS_V1, PATCH_COUNT_V1
        )
        encoded = self.temporal_encoder(tokens, src_key_padding_mask=patch_padding)
        channels = self.temporal_pool(encoded, key_padding_mask=patch_padding).reshape(
            batch, DYNAMIC_CHANNELS_V1, D_MODEL_V1
        )
        ids = torch.arange(DYNAMIC_CHANNELS_V1, device=values.device)
        channels = channels + self.channel_identity(ids)[None]
        normalized = self.channel_norm(channels)
        attended, _ = self.channel_attention(
            normalized, normalized, normalized, need_weights=False
        )
        dynamic = self.channel_pool(channels + attended)
        slow_input = torch.cat(
            (slow_values, slow_available.to(slow_values.dtype)), dim=-1
        )
        fused = self.fusion(
            torch.cat(
                (
                    dynamic,
                    self.slow_encoder(slow_input),
                    self.industry_embedding(industry_id),
                ),
                dim=-1,
            )
        )
        return torch.cat(
            [self.heads[str(horizon)](fused) for horizon in HORIZONS_V1], dim=1
        )


__all__ = [
    "DYNAMIC_CHANNELS_V1",
    "HORIZONS_V1",
    "LOOKBACK_V1",
    "SLOW_CHANNELS_V1",
    "CourageStrictModelError",
    "PatchTSTCourageStrictV1",
]
