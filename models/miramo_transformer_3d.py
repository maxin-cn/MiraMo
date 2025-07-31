# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Dict, Optional, Tuple, Union

import os
import json
import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange, repeat

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.utils import USE_PEFT_BACKEND, logging, scale_lora_layers, unscale_lora_layers
from diffusers.models.attention_processor import (
    Attention,
    AttentionProcessor,
    SanaLinearAttnProcessor2_0,
)
from diffusers.models.embeddings import PatchEmbed, PixArtAlphaTextProjection, TimestepEmbedding, Timesteps
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import RMSNorm

try:
    from models.rope_positional_emb import get_nd_rotary_pos_embed, apply_rotary_emb
except:
    from .rope_positional_emb import get_nd_rotary_pos_embed, apply_rotary_emb

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

class PixArtAlphaCombinedTimestepSizeEmbeddings(nn.Module):
    """
    For PixArt-Alpha.

    Reference:
    https://github.com/PixArt-alpha/PixArt-alpha/blob/0f55e922376d8b797edd44d25d0e7464b260dcab/diffusion/model/nets/PixArtMS.py#L164C9-L168C29
    """

    def __init__(self, embedding_dim, size_emb_dim, use_additional_conditions: bool = False):
        super().__init__()

        self.outdim = size_emb_dim
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

        self.use_additional_conditions = use_additional_conditions
        if use_additional_conditions:
            self.additional_condition_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
            self.resolution_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=size_emb_dim)
            self.aspect_ratio_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=size_emb_dim)

    def forward(self, timestep, resolution, aspect_ratio, batch_size, hidden_dtype):
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=hidden_dtype))  # (N, D)

        if self.use_additional_conditions:
            resolution_emb = self.additional_condition_proj(resolution.flatten()).to(hidden_dtype)
            resolution_emb = self.resolution_embedder(resolution_emb).reshape(batch_size, -1)
            aspect_ratio_emb = self.additional_condition_proj(aspect_ratio.flatten()).to(hidden_dtype)
            aspect_ratio_emb = self.aspect_ratio_embedder(aspect_ratio_emb).reshape(batch_size, -1)
            conditioning = timesteps_emb + torch.cat([resolution_emb, aspect_ratio_emb], dim=1)
        else:
            conditioning = timesteps_emb

        return conditioning

class AdaLayerNormSingle(nn.Module):
    r"""
    Norm layer adaptive layer norm single (adaLN-single).

    As proposed in PixArt-Alpha (see: https://arxiv.org/abs/2310.00426; Section 2.3).

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        use_additional_conditions (`bool`): To use additional conditions for normalization or not.
    """

    def __init__(self, embedding_dim: int, use_additional_conditions: bool = False, use_motion_guidance: bool = False):
        super().__init__()

        self.use_motion_guidance = use_motion_guidance

        self.emb = PixArtAlphaCombinedTimestepSizeEmbeddings(
            embedding_dim, size_emb_dim=embedding_dim // 3, use_additional_conditions=use_additional_conditions
        )

        if use_motion_guidance:
            self.motion_guidance_emb = PixArtAlphaCombinedTimestepSizeEmbeddings(
                embedding_dim, size_emb_dim=embedding_dim // 3, use_additional_conditions=use_additional_conditions
            )

        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, 9 * embedding_dim, bias=True)

    def forward(
        self,
        timestep: torch.Tensor,
        motion_guidance: torch.Tensor = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        batch_size: Optional[int] = None,
        hidden_dtype: Optional[torch.dtype] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # No modulation happening here.
        added_cond_kwargs = added_cond_kwargs or {"resolution": None, "aspect_ratio": None}
        embedded_timestep = self.emb(timestep, **added_cond_kwargs, batch_size=batch_size, hidden_dtype=hidden_dtype)

        if self.use_motion_guidance:
            embedded_motion_guidance = self.motion_guidance_emb(motion_guidance, **added_cond_kwargs, batch_size=batch_size, hidden_dtype=hidden_dtype)
            embedded_timestep = embedded_timestep + embedded_motion_guidance # 0 for image generation

        return self.linear(self.silu(embedded_timestep)), embedded_timestep

class GLUMBConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expand_ratio: float = 4,
        norm_type: Optional[str] = None,
        residual_connection: bool = True,
    ) -> None:
        super().__init__()

        hidden_channels = int(expand_ratio * in_channels)
        self.norm_type = norm_type
        self.residual_connection = residual_connection

        self.nonlinearity = nn.SiLU()
        self.conv_inverted = nn.Conv2d(in_channels, hidden_channels * 2, 1, 1, 0)
        self.conv_depth = nn.Conv2d(hidden_channels * 2, hidden_channels * 2, 3, 1, 1, groups=hidden_channels * 2)
        self.conv_point = nn.Conv2d(hidden_channels, out_channels, 1, 1, 0, bias=False)

        self.norm = None
        if norm_type == "rms_norm":
            self.norm = RMSNorm(out_channels, eps=1e-5, elementwise_affine=True, bias=True)


    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.residual_connection:
            residual = hidden_states

        hidden_states = self.conv_inverted(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.conv_depth(hidden_states)
        hidden_states, gate = torch.chunk(hidden_states, 2, dim=1)
        hidden_states = hidden_states * self.nonlinearity(gate)

        hidden_states = self.conv_point(hidden_states)

        if self.norm_type == "rms_norm":
            # move channel to the last dimension so we apply RMSnorm across channel dimension
            hidden_states = self.norm(hidden_states.movedim(1, -1)).movedim(-1, 1)

        if self.residual_connection:
            hidden_states = hidden_states + residual

        return hidden_states


class MiraMoModulatedNorm(nn.Module):
    def __init__(self, dim: int, elementwise_affine: bool = False, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(
        self, hidden_states: torch.Tensor, temb: torch.Tensor, scale_shift_table: torch.Tensor, num_frames: int = 1
    ) -> torch.Tensor:
        hidden_states = self.norm(hidden_states)
        shift, scale = (scale_shift_table[None] + temb[:, None].to(scale_shift_table.device)).chunk(2, dim=1)
        if num_frames > 1:
            scale = repeat(scale, "b m d -> (b f) m d", f=num_frames).contiguous()
            shift = repeat(shift, "b m d -> (b f) m d", f=num_frames).contiguous()
        hidden_states = hidden_states * (1 + scale) + shift
        return hidden_states


class MiraMoCombinedTimestepGuidanceEmbeddings(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

        self.guidance_condition_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.guidance_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, 6 * embedding_dim, bias=True)

    def forward(self, timestep: torch.Tensor, guidance: torch.Tensor = None, hidden_dtype: torch.dtype = None):
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=hidden_dtype))  # (N, D)

        guidance_proj = self.guidance_condition_proj(guidance)
        guidance_emb = self.guidance_embedder(guidance_proj.to(dtype=hidden_dtype))
        conditioning = timesteps_emb + guidance_emb

        return self.linear(self.silu(conditioning)), conditioning


class MiraMoAttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("MiraMoAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states
    
class RopeLinearAttnProcessor2_0:
    r"""
    Processor implementing formula7-based linear attention
    """

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        freqs_cis: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        original_dtype = hidden_states.dtype
        
        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        batch_size, sequence_length, _ = encoder_hidden_states.shape

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        query = query.unflatten(2, (attn.heads, -1)) # [B, seq_len, num_heads, head_dim]
        key = key.unflatten(2, (attn.heads, -1)) # [B, seq_len, num_heads, head_dim]
        value = value.unflatten(2, (attn.heads, -1)) # [B, seq_len, num_heads, head_dim]

        # normalize query and key
        query = F.normalize(query, p=2, dim=-1)  # q_i / ||q_i||
        key = F.normalize(key, p=2, dim=-1)      # k_j / ||k_j||

        if freqs_cis is not None:
            # xq should have this shape
            # bsz, seqlen, (self.n_local_heads, self.head_dim)
            query, key = apply_rotary_emb(xq=query, xk=key, freqs_cis=freqs_cis)

        query = query.permute(0, 2, 3, 1).contiguous() # [B, num_heads, head_dim, seq_len]
        key = key.permute(0, 2, 1, 3).contiguous() # [B, num_heads, seq_len, head_dim]
        value = value.permute(0, 2, 3, 1).contiguous() # [B, num_heads, head_dim, seq_len]

        # precompute global statistics
        S_V = torch.matmul(value, key)  # ∑(k_j/||k_j|| ⊗ V_j)
        S_1 = torch.sum(key, dim=-2, keepdim=True)  # ∑(k_j/||k_j||)

        # Merge matrix multiplications
        combined_S = torch.cat([S_V, S_1], dim=2) 
        combined_result = torch.matmul(combined_S, query) 

        # Split results
        S_V_query = combined_result[:, :, :-1, :] 
        S_1_query = combined_result[:, :, -1:, :] 

        # Compute numerator and denominator
        sum_v = torch.sum(value, dim=-1, keepdim=True)
        numerator = sum_v + S_V_query
        denominator = sequence_length + S_1_query  
        
        # 最终输出
        hidden_states = (numerator / denominator.clamp(min=1e-6)).flatten(1, 2).transpose(1, 2)
        hidden_states = hidden_states.to(original_dtype)

        # 输出投影层
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states

class MiraMoTransformerBlock(nn.Module):
    r"""
    Transformer block introduced in [MiraMo](https://huggingface.co/papers/2410.10629).
    """

    def __init__(
        self,
        dim: int = 2240,
        num_attention_heads: int = 70,
        attention_head_dim: int = 32,
        dropout: float = 0.0,
        num_cross_attention_heads: Optional[int] = 20,
        cross_attention_head_dim: Optional[int] = 112,
        cross_attention_dim: Optional[int] = 2240,
        attention_bias: bool = True,
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-6,
        attention_out_bias: bool = True,
        mlp_ratio: float = 2.5,
        qk_norm: Optional[str] = None,
    ) -> None:
        super().__init__()

        # 1. Self Attention
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=norm_eps)
        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            kv_heads=num_attention_heads if qk_norm is not None else None,
            qk_norm=qk_norm,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=None,
            processor=SanaLinearAttnProcessor2_0(),
        )

        # 1.5 Temporal Attention
        self.norm1_temporal = nn.LayerNorm(dim, elementwise_affine=False, eps=norm_eps)
        self.attn_temporal = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            kv_heads=num_attention_heads if qk_norm is not None else None,
            qk_norm=qk_norm,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=None,
            processor=RopeLinearAttnProcessor2_0(),
        )

        # 2. Cross Attention
        if cross_attention_dim is not None:
            self.norm2 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)
            self.attn2 = Attention(
                query_dim=dim,
                qk_norm=qk_norm,
                kv_heads=num_cross_attention_heads if qk_norm is not None else None,
                cross_attention_dim=cross_attention_dim,
                heads=num_cross_attention_heads,
                dim_head=cross_attention_head_dim,
                dropout=dropout,
                bias=True,
                out_bias=attention_out_bias,
                processor=MiraMoAttnProcessor2_0(),
            )

        # 3. Feed-forward
        self.ff = GLUMBConv(dim, dim, mlp_ratio, norm_type=None, residual_connection=False)

        self.scale_shift_table = nn.Parameter(torch.randn(6, dim) / dim**0.5)
        self.scale_shift_table_attn3d = nn.Parameter(torch.randn(3, dim) / dim**0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        height: int = None,
        width: int = None,
        num_frames: int = None,
        freqs_cis: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, sequence_len, _ = hidden_states.shape

        # 1. Modulation
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp, shift_msa_temporal, scale_msa_temporal, gate_msa_temporal = (
            torch.cat([self.scale_shift_table[None], self.scale_shift_table_attn3d[None]], dim=1) + timestep.reshape(batch_size // num_frames, 9, -1)
        ).chunk(9, dim=1)

        # 2. Self Attention
        norm_hidden_states = self.norm1(hidden_states)
        if num_frames > 1:
            scale_msa = repeat(scale_msa, "b m d -> (b f) m d", f=num_frames).contiguous()
            shift_msa = repeat(shift_msa, "b m d -> (b f) m d", f=num_frames).contiguous()
        norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
        norm_hidden_states = norm_hidden_states.to(hidden_states.dtype)

        attn_output = self.attn1(norm_hidden_states)
        if num_frames > 1:
            gate_msa = repeat(gate_msa, "b m d -> (b f) m d", f=num_frames).contiguous()
        hidden_states = hidden_states + gate_msa * attn_output

        # 2.5 Temporal Attention
        hidden_states = rearrange(hidden_states, "(b f) t d -> (b t) f d", f=num_frames).contiguous()
        norm_hidden_states_temporal = self.norm1_temporal(hidden_states)
        scale_msa_temporal = repeat(scale_msa_temporal, "b m d -> (b t) m d", t=sequence_len).contiguous()
        shift_msa_temporal = repeat(shift_msa_temporal, "b m d -> (b t) m d", t=sequence_len).contiguous()
        gate_msa_temporal = repeat(gate_msa_temporal, "b m d -> (b t) m d", t=sequence_len).contiguous()
        norm_hidden_states_temporal = norm_hidden_states_temporal * (1 + scale_msa_temporal) + shift_msa_temporal
        norm_hidden_states_temporal = norm_hidden_states_temporal.to(hidden_states.dtype)

        attn_output_temporal = self.attn_temporal(norm_hidden_states_temporal, freqs_cis=freqs_cis)
        hidden_states = hidden_states + gate_msa_temporal * attn_output_temporal
        hidden_states = rearrange(hidden_states, "(b t) f d -> (b f) t d", f=num_frames, t=sequence_len).contiguous()

        # 3. Cross Attention
        if num_frames > 1:
            encoder_hidden_states = repeat(encoder_hidden_states, "b t d -> (b f) t d", f=num_frames).contiguous()
            if encoder_attention_mask is not None:
                encoder_attention_mask = repeat(encoder_attention_mask, "b m n -> (b f) m n", f=num_frames).contiguous()
        if self.attn2 is not None:
            attn_output = self.attn2(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
            )
            hidden_states = attn_output + hidden_states

        # 4. Feed-forward
        norm_hidden_states = self.norm2(hidden_states)
        if num_frames > 1:
            scale_mlp = repeat(scale_mlp, "b m d -> (b f) m d", f=num_frames).contiguous()
            shift_mlp = repeat(shift_mlp, "b m d -> (b f) m d", f=num_frames).contiguous()
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp

        norm_hidden_states = norm_hidden_states.unflatten(1, (height, width)).permute(0, 3, 1, 2).contiguous() # (b f) d h w
        ff_output = self.ff(norm_hidden_states) # (b f) d h w; conv2d, b c h w
        ff_output = ff_output.flatten(2, 3).permute(0, 2, 1).contiguous()
        if num_frames > 1:
            gate_mlp = repeat(gate_mlp, "b m d -> (b f) m d", f=num_frames).contiguous()
        hidden_states = hidden_states + gate_mlp * ff_output

        return hidden_states


class MiraMoTransformer3DModel(ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin):
    r"""
    A 3D Transformer model introduced in MiraMo.

    Args:
        in_channels (`int`, defaults to `32`):
            The number of channels in the input.
        out_channels (`int`, *optional*, defaults to `32`):
            The number of channels in the output.
        num_attention_heads (`int`, defaults to `70`):
            The number of heads to use for multi-head attention.
        attention_head_dim (`int`, defaults to `32`):
            The number of channels in each head.
        num_layers (`int`, defaults to `20`):
            The number of layers of Transformer blocks to use.
        num_cross_attention_heads (`int`, *optional*, defaults to `20`):
            The number of heads to use for cross-attention.
        cross_attention_head_dim (`int`, *optional*, defaults to `112`):
            The number of channels in each head for cross-attention.
        cross_attention_dim (`int`, *optional*, defaults to `2240`):
            The number of channels in the cross-attention output.
        caption_channels (`int`, defaults to `2304`):
            The number of channels in the caption embeddings.
        mlp_ratio (`float`, defaults to `2.5`):
            The expansion ratio to use in the GLUMBConv layer.
        dropout (`float`, defaults to `0.0`):
            The dropout probability.
        attention_bias (`bool`, defaults to `False`):
            Whether to use bias in the attention layer.
        sample_size (`int`, defaults to `32`):
            The base size of the input latent.
        patch_size (`int`, defaults to `1`):
            The size of the patches to use in the patch embedding layer.
        norm_elementwise_affine (`bool`, defaults to `False`):
            Whether to use elementwise affinity in the normalization layer.
        norm_eps (`float`, defaults to `1e-6`):
            The epsilon value for the normalization layer.
        qk_norm (`str`, *optional*, defaults to `None`):
            The normalization to use for the query and key.
        timestep_scale (`float`, defaults to `1.0`):
            The scale to use for the timesteps.
    """

    _supports_gradient_checkpointing = True
    _no_split_modules = ["MiraMoTransformerBlock", "PatchEmbed", "MiraMoModulatedNorm"]
    _skip_layerwise_casting_patterns = ["patch_embed", "norm"]

    @register_to_config
    def __init__(
        self,
        in_channels: int = 32,
        out_channels: Optional[int] = 32,
        num_attention_heads: int = 70,
        attention_head_dim: int = 32,
        num_layers: int = 20,
        num_cross_attention_heads: Optional[int] = 20,
        cross_attention_head_dim: Optional[int] = 112,
        cross_attention_dim: Optional[int] = 2240,
        caption_channels: int = 2304,
        mlp_ratio: float = 2.5,
        dropout: float = 0.0,
        attention_bias: bool = False,
        sample_size: int = 32,
        patch_size: int = 1,
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-6,
        interpolation_scale: Optional[int] = None,
        guidance_embeds: bool = False,
        guidance_embeds_scale: float = 0.1,
        qk_norm: Optional[str] = None,
        timestep_scale: float = 1.0,
        use_motion_guidance: bool = False,
    ) -> None:
        super().__init__()

        out_channels = out_channels or in_channels
        inner_dim = num_attention_heads * attention_head_dim

        # 1. Patch Embedding
        self.patch_embed = PatchEmbed(
            height=sample_size,
            width=sample_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=inner_dim,
            interpolation_scale=interpolation_scale,
            pos_embed_type="sincos" if interpolation_scale is not None else None,
        )

        # 2. Additional condition embeddings
        if guidance_embeds:
            self.time_embed = MiraMoCombinedTimestepGuidanceEmbeddings(inner_dim)
        else:
            self.time_embed = AdaLayerNormSingle(inner_dim, use_motion_guidance=use_motion_guidance)
            self.use_motion_guidance = use_motion_guidance
            print('use_motion_guidance', use_motion_guidance)

        self.caption_projection = PixArtAlphaTextProjection(in_features=caption_channels, hidden_size=inner_dim)
        self.caption_norm = RMSNorm(inner_dim, eps=1e-5, elementwise_affine=True)

        # 3. Transformer blocks
        self.transformer_blocks = nn.ModuleList(
            [
                MiraMoTransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout=dropout,
                    num_cross_attention_heads=num_cross_attention_heads,
                    cross_attention_head_dim=cross_attention_head_dim,
                    cross_attention_dim=cross_attention_dim,
                    attention_bias=attention_bias,
                    norm_elementwise_affine=norm_elementwise_affine,
                    norm_eps=norm_eps,
                    mlp_ratio=mlp_ratio,
                    qk_norm=qk_norm,
                )
                for _ in range(num_layers)
            ]
        )

        # 3.5 Define freq cis
        rope_dim_list = [32]
        rope_sizes = [64]
        cos_freq, sin_freq = get_nd_rotary_pos_embed(rope_dim_list=rope_dim_list,
                                                     start=rope_sizes,
                                                     use_real=True,
                                                     theta_rescale_factor=1,
                                                    #  theta=256.0,
                                                     )
        self.register_buffer("cos_freq", cos_freq, persistent=False)
        self.register_buffer("sin_freq", sin_freq, persistent=False)
        
        # 4. Output blocks
        self.scale_shift_table = nn.Parameter(torch.randn(2, inner_dim) / inner_dim**0.5)
        self.norm_out = MiraMoModulatedNorm(inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out = nn.Linear(inner_dim, patch_size * patch_size * out_channels)

        self.gradient_checkpointing = False

    @property
    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.attn_processors
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor()

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_attn_processor
    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        guidance: Optional[torch.Tensor] = None,
        motion_bins: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ) -> Union[Tuple[torch.Tensor, ...], Transformer2DModelOutput]:
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                )

        # ensure attention_mask is a bias, and give it a singleton query_tokens dimension.
        #   we may have done this conversion already, e.g. if we came here via UNet2DConditionModel#forward.
        #   we can tell by counting dims; if ndim == 2: it's a mask rather than a bias.
        # expects mask of shape:
        #   [batch, key_tokens]
        # adds singleton query_tokens dimension:
        #   [batch,                    1, key_tokens]
        # this helps to broadcast it as a bias over attention scores, which will be in one of the following shapes:
        #   [batch,  heads, query_tokens, key_tokens] (e.g. torch sdp attn)
        #   [batch * heads, query_tokens, key_tokens] (e.g. xformers or classic attn)
        if attention_mask is not None and attention_mask.ndim == 2:
            # assume that mask is expressed as:
            #   (1 = keep,      0 = discard)
            # convert mask into a bias that can be added to attention scores:
            #       (keep = +0,     discard = -10000.0)
            attention_mask = (1 - attention_mask.to(hidden_states.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # convert encoder_attention_mask to a bias the same way we do for attention_mask
        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
            encoder_attention_mask = (1 - encoder_attention_mask.to(hidden_states.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        # 1. Input
        batch_size, num_frames, num_channels, height, width = hidden_states.shape
        p = self.config.patch_size
        post_patch_height, post_patch_width = height // p, width // p

        if num_frames > 1:
            cos_freq = self.cos_freq[:num_frames, ...]
            sin_freq = self.sin_freq[:num_frames, ...]
            freqs_cis = (cos_freq, sin_freq)
        else:
            freqs_cis = None

        hidden_states = rearrange(hidden_states, "b f c h w -> (b f) c h w").contiguous()
        hidden_states = self.patch_embed(hidden_states)
        if guidance is not None:
            timestep, embedded_timestep = self.time_embed(
                timestep, guidance=guidance, hidden_dtype=hidden_states.dtype
            )
        else:
            if self.use_motion_guidance:
                timestep, embedded_timestep = self.time_embed(
                    timestep, motion_guidance=motion_bins, batch_size=batch_size, hidden_dtype=hidden_states.dtype
                )
            else:
                timestep, embedded_timestep = self.time_embed(
                    timestep, batch_size=batch_size, hidden_dtype=hidden_states.dtype
                )

        encoder_hidden_states = self.caption_projection(encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states.view(batch_size, -1, hidden_states.shape[-1])

        encoder_hidden_states = self.caption_norm(encoder_hidden_states)

        # 2. Transformer blocks
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for block in self.transformer_blocks:
                hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    timestep,
                    post_patch_height,
                    post_patch_width,
                    num_frames,
                    freqs_cis,
                )

        else:
            for block in self.transformer_blocks:
                hidden_states = block(
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    timestep,
                    post_patch_height,
                    post_patch_width,
                    num_frames,
                    freqs_cis,
                )

        # 3. Normalization
        hidden_states = self.norm_out(hidden_states, embedded_timestep, self.scale_shift_table, num_frames=num_frames)

        hidden_states = self.proj_out(hidden_states)

        output = rearrange(
            hidden_states,
            '(b f) (h w) (p1 p2 c) -> b f c (h p1) (w p2)',
            b=batch_size,
            f=num_frames,
            h=post_patch_height,
            w=post_patch_width,
            p1=self.config.patch_size,
            p2=self.config.patch_size
        ).contiguous()

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)

    @classmethod
    def from_pretrained_2d(cls, 
                           pretrained_model_name_or_path, 
                           subfolder=None, 
                           **kwargs):
        if subfolder is not None:
            pretrained_model_path = os.path.join(pretrained_model_name_or_path, subfolder)

        config_file = os.path.join(pretrained_model_path, 'config.json')
        if not os.path.isfile(config_file):
            raise RuntimeError(f"{config_file} does not exist")
        with open(config_file, "r") as f:
            config = json.load(f)

        config["patch_size"] = 2
        config["in_channels"] = 4
        config["out_channels"] = 4

        model = cls.from_config(config, **kwargs)
        
        model_files = [
            os.path.join(pretrained_model_path, 'diffusion_pytorch_model.bin'),
            os.path.join(pretrained_model_path, 'diffusion_pytorch_model.safetensors')
        ]

        model_file = None

        for fp in model_files:
            if os.path.exists(fp):
                model_file = fp

        if not model_file:
            raise RuntimeError(f"{model_file} does not exist")

        if model_file.split(".")[-1] == "safetensors":
            from safetensors import safe_open
            state_dict = {}
            with safe_open(model_file, framework="pt", device="cpu") as f:
                for key in f.keys():
                    state_dict[key] = f.get_tensor(key)
        else:
            state_dict = torch.load(model_file, map_location="cpu")
        
        # for k, v in model.state_dict().items():
        #     if 'motion_guidance_emb' in k:
        #         state_dict.update({k: v})
        #     if 'time_embed.linear' in k:
        #         original_six_weight = state_dict[k]
        #         # print("orinal_six_weight", original_six_weight.shape)
        #         additional_weight = original_six_weight[:3 * 1152, ...]
        #         new_weight = torch.cat([original_six_weight, additional_weight], dim=0)
        #         state_dict.update({k: new_weight})
        #     if 'scale_shift_table_attn3d' in k: # for norm_elementwise_affine
        #         state_dict.update({k: v})
        #     if 'attn_temporal' in k: # for norm_elementwise_affine
        #         kn = k.replace("attn_temporal", "attn1")
        #         attn1_weight = state_dict[kn]
        #         state_dict.update({k: attn1_weight})
        #     if "patch_embed" in k:
        #         state_dict.update({k: v})
            
        #     if "proj_out" in k:
        #         state_dict.update({k: v})

        # model.load_state_dict(state_dict)

        return model
    
if __name__ == "__main__":
    pre_trained_model_name_or_path = "/mnt/hwfile/gcc/maxin/work/pretrained/Efficient-Large-Model/MiraMo_600M_512px_diffusers"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch_dtype = torch.float16

    model = MiraMoTransformer3DModel.from_pretrained_2d(pretrained_model_name_or_path=pre_trained_model_name_or_path, subfolder="transformer", torch_dtype=torch_dtype).to(device)
    model.train()
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    hidden_states = torch.randn(2, 16, 32, 8, 8).to(device)
    encoder_hidden_states = torch.randn(2, 72, 2304).to(device)
    encoder_attention_mask = torch.zeros(2, 72).to(device)
    encoder_attention_mask[0, :32] = 1
    encoder_attention_mask[1, :55] = 1

    timestep = torch.tensor([0, 1]).to(device)

    out = model(hidden_states, encoder_hidden_states, timestep, encoder_attention_mask=encoder_attention_mask)
    print(out[0].shape)