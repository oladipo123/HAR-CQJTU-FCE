from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

from torchtsmixer.layers import MixerLayer, TimeBatchNorm2d, feature_to_time, time_to_feature

def gelu(x):
    "Implementation of the gelu activation function by Hugging Face"
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


# class GELULeakyReLU(torch.nn.Module):
#     def __init__(self):
#         super(GELULeakyReLU, self).__init__()
#
#     def forward(self, x):
#         # 计算 GELU 激活函数的输出
#         gelu_x = 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * x ** 3)))
#
#         # 对负值区间使用 GELU(x) 作为斜率
#         negative = gelu_x * x
#         positive = x  # 对于正值部分，直接使用 Leaky ReLU 或 ReLU
#
#         # 结合正值和负值部分
#         output = torch.where(x < 0, negative, positive)
#         return output
class GELUReLUNonlinear(nn.Module):
    def __init__(self):
        super(GELUReLUNonlinear, self).__init__()

    def forward(self, x):
        sigmoid_part = torch.sigmoid(x)  # 非线性权重调整
        elu_part =  F.elu(x)
        gelu_part = F.gelu(x)
        return sigmoid_part * elu_part + (1 - sigmoid_part) * gelu_part
        # return (1 - sigmoid_part) * elu_part + sigmoid_part * gelu_part


class TSMixer(nn.Module):
    """TSMixer model for time series forecasting.

    This model uses a series of mixer layers to process time series data,
    followed by a linear transformation to project the output to the desired
    prediction length.

    Attributes:
        mixer_layers: Sequential container of mixer layers.
        temporal_projection: Linear layer for temporal projection.

    Args:
        sequence_length: Length of the input time series sequence.
        prediction_length: Desired length of the output prediction sequence.
        input_channels: Number of input channels.
        output_channels: Number of output channels. Defaults to None.
        activation_fn: Activation function to use. Defaults to "relu".
        num_blocks: Number of mixer blocks. Defaults to 2.
        dropout_rate: Dropout rate for regularization. Defaults to 0.1.
        ff_dim: Dimension of feedforward network inside mixer layer. Defaults to 64.
        normalize_before: Whether to apply layer normalization before or after mixer layer.
        norm_type: Type of normalization to use. "batch" or "layer". Defaults to "batch".
    """

    def __init__(
        self,
        cfg ,
        sequence_length: int,
        prediction_length: int,
        input_channels: int,
        output_channels: int = None,
        activation_fn: str = "leaky_relu",
        num_blocks: int = 2,
        dropout_rate: float = 0.1,
        ff_dim: int = 144,
        normalize_before: bool = True,
        norm_type: str = "layer",

    ):
        super().__init__()

        # Transform activation_fn to callable
        activation_fn = getattr(F, activation_fn)
        print(f" the activation is {activation_fn}")
        gelu_leaky_relu = GELUReLUNonlinear()

        # Transform norm_type to callable
        assert norm_type in {
            "batch",
            "layer",
        }, f"Invalid norm_type: {norm_type}, must be one of batch, layer."
        norm_type = TimeBatchNorm2d if norm_type == "batch" else nn.LayerNorm

        # Build mixer layers
        self.mixer_layers = self._build_mixer(
            num_blocks,
            input_channels,
            output_channels,
            ff_dim=ff_dim,
            activation_fn=gelu_leaky_relu,
            dropout_rate=dropout_rate,
            sequence_length=sequence_length,
            normalize_before=normalize_before,
            norm_type=norm_type,
        )
        self.embed = Embeddings(cfg)
        self.attn = MultiHeadedSelfAttention(cfg)
        self.n_layers = cfg.n_layers
        # Temporal projection layer
        self.temporal_projection = nn.Linear(sequence_length, prediction_length)

    def _build_mixer(
        self, num_blocks: int, input_channels: int, output_channels: int, **kwargs
    ):
        """Build the mixer blocks for the model.

        Args:
            num_blocks (int): Number of mixer blocks to be built.
            input_channels (int): Number of input channels for the first block.
            output_channels (int): Number of output channels for the last block.
            **kwargs: Additional keyword arguments for mixer layer configuration.

        Returns:
            nn.Sequential: Sequential container of mixer layers.
        """
        output_channels = output_channels if output_channels is not None else input_channels
        channels = [input_channels] * (num_blocks - 1) + [output_channels]

        return nn.Sequential(
            *[
                MixerLayer(input_channels=in_ch, output_channels=out_ch, **kwargs)
                for in_ch, out_ch in zip(channels[:-1], channels[1:])
            ]
        )

    def forward(self, x_hist: torch.Tensor) -> torch.Tensor:
        """Forward pass of the TSMixer model.

        Args:
            x_hist (torch.Tensor): Input time series tensor.

        Returns:
            torch.Tensor: The output tensor after processing by the model.
        """
        x_hist = self.embed(x_hist)
        # print(self.n_layers)
        for _ in range(self.n_layers):
            x_hist = self.attn(x_hist)
        # for _ in range(self.n_layers):
        # print(f"the x_hist is {x_hist.shape}")
            x = self.mixer_layers(x_hist)

            x_temp = feature_to_time(x)
            x_temp = self.temporal_projection(x_temp)
            x = time_to_feature(x_temp)
        return x

class MultiHeadedSelfAttention(nn.Module):
    """ Multi-Headed Dot Product Attention """
    def __init__(self, cfg):
        super().__init__()
        self.proj_q = nn.Linear(cfg.hidden, cfg.hidden)
        self.proj_k = nn.Linear(cfg.hidden, cfg.hidden)
        self.proj_v = nn.Linear(cfg.hidden, cfg.hidden)
        # self.drop = nn.Dropout(cfg.p_drop_attn)
        self.scores = None # for visualization
        self.n_heads = cfg.n_heads

    def forward(self, x):
        """
        x, q(query), k(key), v(value) : (B(batch_size), S(seq_len), D(dim))
        * split D(dim) into (H(n_heads), W(width of head)) ; D = H * W
        """
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)
        q, k, v = (split_last(x, (self.n_heads, -1)).transpose(1, 2)
                   for x in [q, k, v])
        # (B, H, S, W) @ (B, H, W, S) -> (B, H, S, S) -softmax-> (B, H, S, S)
        scores = q @ k.transpose(-2, -1) / np.sqrt(k.size(-1))
        #scores = self.drop(F.softmax(scores, dim=-1))
        scores = F.softmax(scores, dim=-1)
        # (B, H, S, S) @ (B, H, S, W) -> (B, H, S, W) -trans-> (B, S, H, W)
        h = (scores @ v).transpose(1, 2).contiguous()
        # -merge-> (B, S, D)
        h = merge_last(h, 2)
        self.scores = scores
        return h


def split_last(x, shape):
    "split the last dimension to given shape"
    shape = list(shape)
    assert shape.count(-1) <= 1
    if -1 in shape:
        shape[shape.index(-1)] = x.size(-1) // -np.prod(shape)
    return x.view(*x.size()[:-1], *shape)


def merge_last(x, n_dims):
    "merge the last n_dims to a dimension"
    s = x.size()
    assert n_dims > 1 and n_dims < len(s)
    return x.view(*s[:-n_dims], -1)

class Embeddings(nn.Module):

    def __init__(self, cfg, pos_embed=None):
        super().__init__()

        # factorized embedding
        self.lin = nn.Linear(cfg.feature_num, cfg.hidden)
        if pos_embed is None:
            self.pos_embed = nn.Embedding(cfg.seq_len, cfg.hidden) # position embedding
        else:
            self.pos_embed = pos_embed

        self.norm = LayerNorm(cfg)
        self.emb_norm = cfg.emb_norm

    def forward(self, x):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long, device=x.device)
        pos = pos.unsqueeze(0).expand(x.size(0), seq_len) # (S,) -> (B, S)

        # factorized embedding

        e = self.lin(x)
        # print(f"emb_norm is {self.emb_norm}")
        if self.emb_norm:
            e = self.norm(e)
        e = e + self.pos_embed(pos)
        return self.norm(e)

class LayerNorm(nn.Module):
    "A layernorm module in the TF style (epsilon inside the square root)."
    def __init__(self, cfg, variance_epsilon=1e-12):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(cfg.hidden), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(cfg.hidden), requires_grad=True)
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta

if __name__ == "__main__":
    m = TSMixer(120, 120, 6, output_channels=72)
    x = torch.randn(128, 120, 6)
    y = m(x)
    print(y.shape)
