from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from layers.Embed import DataEmbedding
from layers.Graph_Enhanced_Module import GraphBlock, Attention_Block, Predict
from layers.Multi_Scale_Module import Multi_Scale_Module
from torch import Tensor
import copy



def FFT_for_Period(x, k=3):
    # x: [B, T] or [B, T, C]
    xf = torch.fft.rfft(x, dim=1)

    if x.ndim == 3:
        # x has shape [B, T, C]
        frequency_list = abs(xf).mean(dim=(0, 2))  # Shape [N]
    elif x.ndim == 2:
        # x has shape [B, T]
        frequency_list = abs(xf).mean(dim=0)  # Shape [N]
    else:
        raise ValueError("Input x should be 2D or 3D tensor")

    frequency_list[0] = 0  # Zero out the DC component
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list

    # Compute scale_weight
    if x.ndim == 3:
        scale_weight = abs(xf).mean(dim=2)[:, top_list]  # [B, N], mean over channels
    elif x.ndim == 2:
        scale_weight = abs(xf)[:, top_list]  # [B, N]
    # print("period", period)
    return period, scale_weight


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        # x: [B, T, C]
        padding_size = self.kernel_size // 2
        if self.kernel_size % 2 == 0:
            # Even kernel size
            left_padding = padding_size - 1
            right_padding = padding_size
        else:
            # Odd kernel size
            left_padding = padding_size
            right_padding = padding_size
        # padding on both ends of time series
        front = x[:, 0:1, :].repeat(1, left_padding, 1)
        end = x[:, -1:, :].repeat(1, right_padding, 1)
        x = torch.cat([front, x, end], dim=1)
        x = x.permute(0, 2, 1)  # [B, C, T]
        x = F.avg_pool1d(x, kernel_size=self.kernel_size, stride=self.stride)
        x = x.permute(0, 2, 1)  # [B, T, C]
        return x

class series_decomp(nn.Module):
    """
    Series decomposition block with multiple levels of decomposition
    """

    def __init__(self, kernel_size_list):
        super(series_decomp, self).__init__()
        self.moving_avgs = nn.ModuleList([
            moving_avg(kernel_size=ks, stride=1) for ks in kernel_size_list
        ])

    def forward(self, x):
        res = x
        moving_means = []
        for moving_avg_layer in self.moving_avgs:
            moving_mean = moving_avg_layer(res)
            moving_means.append(moving_mean)
            res = res - moving_mean
        # res is the final residual (seasonal component)
        return moving_means, res

class TrendBlock(nn.Module):
    def __init__(self, configs):
        super(TrendBlock, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        # Since use_gconv=True for trend components
        self.gconv = GraphBlock(configs.enc_in, configs.d_model, configs.conv_channel, configs.skip_channel,
                                configs.gcn_depth, configs.dropout, configs.propalpha, configs.seq_len,
                                configs.node_dim)

        # Since use_attention=False, we use a Linear layer
        self.linear0 = nn.Linear(configs.d_model, configs.d_model)
        self.norm = nn.LayerNorm(configs.d_model)
        self.gelu = nn.GELU()

    def forward(self, x):
        # x: [B, T, N]
        x_gconv = self.gconv(x)

        # Apply Linear layer
        out = self.norm(self.linear0(x_gconv))
        out = self.gelu(out)

        # Residual connection
        out = out + x
        return out

class SeasonalBlock(nn.Module):
    def __init__(self, configs, max_seq_len: Optional[int] = 1024, d_k: Optional[int] = None, d_v: Optional[int] = None,
                 norm: str = 'BatchNorm', attn_dropout: float = 0.,
                 act: str = "gelu", key_padding_mask: bool = 'auto', padding_var: Optional[int] = None,
                 attn_mask: Optional[Tensor] = None, res_attention: bool = True,
                 pre_norm: bool = False, store_attn: bool = False, pe: str = 'zeros', learn_pe: bool = True,
                 pretrain_head: bool = False, head_type='flatten', verbose: bool = False, **kwargs):

        super().__init__()

        # load parameters
        c_in = configs.target_num
        context_window = configs.seq_len
        target_window = configs.pred_len

        n_layers = configs.e_layers
        n_heads = configs.n_heads
        d_model = configs.d_model
        d_ff = configs.d_ff
        dropout = configs.dropout
        fc_dropout = configs.fc_dropout
        head_dropout = configs.head_dropout

        individual = configs.individual

        add = configs.add
        wo_conv = configs.wo_conv
        serial_conv = configs.serial_conv

        kernel_list = configs.kernel_list
        patch_len = configs.patch_len
        period = configs.period
        scale_weight = configs.scale_weight
        stride = configs.stride

        padding_patch = configs.padding_patch

        revin = configs.revin
        affine = configs.affine
        subtract_last = configs.subtract_last

        # model
        self.model = Multi_Scale_Module(c_in=c_in, context_window=context_window, target_window=target_window,
                                           wo_conv=wo_conv, serial_conv=serial_conv, add=add,
                                           patch_len=patch_len, kernel_list=kernel_list, period=period, scale_weight=scale_weight,stride=stride,
                                           max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                           n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                                           attn_dropout=attn_dropout,
                                           dropout=dropout, act=act, key_padding_mask=key_padding_mask,
                                           padding_var=padding_var,
                                           attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm,
                                           store_attn=store_attn,
                                           pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout,
                                           head_dropout=head_dropout,
                                           padding_patch=padding_patch,
                                           pretrain_head=pretrain_head, head_type=head_type, individual=individual,
                                           revin=revin, affine=affine,
                                           subtract_last=subtract_last, verbose=verbose, **kwargs)

    def forward(self, x):  # x: [Batch, Input length, Channel]
        x = x.permute(0, 2, 1)  # x: [Batch, Channel, Input length]
        # Update the period and scale_weight in the internal model
        x = self.model(x)
        x = x.permute(0, 2, 1)  # x: [Batch, Output length, Channel]
        return x

# (Imports and other classes remain unchanged)

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.k = configs.top_k

        # Define the number of target features
        self.target_num = configs.target_num

        # Decomposition
        kernel_size_list = [48, 32, 24]  # [48, 32, 24, 8, 4]
        self.decomp = series_decomp(kernel_size_list)

        # Get number of input features
        c_in = configs.enc_in  # Total number of input features
        configs.c_out = c_in  # Set c_out to c_in

        # Update enc_in for trend and seasonal embeddings
        self.c_in = c_in
        self.num_trends = len(kernel_size_list)  # Number of trend components

        # Embeddings for trends and seasonal components
        self.enc_embedding_trends = DataEmbedding(c_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
        self.enc_embedding_seasonal = DataEmbedding(self.target_num, configs.d_model,
                                                    configs.embed, configs.freq, configs.dropout)

        # Models for trends and seasonal components
        self.layer = configs.e_layers

        # Trend components: share the same model
        self.model_trends = nn.ModuleList([
            TrendBlock(configs)
            for _ in range(self.layer)
        ])

        # Seasonal component: uses attention and scale
        # Update configs for seasonal block
        seasonal_configs = copy.deepcopy(configs)
        seasonal_configs.enc_in = self.target_num
        self.model_seasonal = SeasonalBlock(seasonal_configs)

        # Layer norms
        self.layer_norm_trends = nn.ModuleList([
            nn.LayerNorm(configs.d_model)
            for _ in range(self.layer)
        ])

        # Projection layers
        self.projection_trends = nn.Linear(configs.d_model, c_in, bias=True)
        self.projection_seasonal = nn.Linear(configs.d_model, self.target_num, bias=True)

        # Prediction layers
        self.seq2pred_trends = Predict(configs.individual, c_in,
                                       configs.seq_len, configs.pred_len, configs.dropout)
        self.seq2pred_seasonal = Predict(configs.individual, self.target_num,
                                         configs.seq_len, configs.pred_len, configs.dropout)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # Standardize the entire input data
        means = x_enc.mean(dim=1, keepdim=True).detach()
        x_normalized = x_enc - means
        stdev = torch.sqrt(torch.var(x_normalized, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_normalized /= stdev

        # Decompose the normalized data
        moving_means, res = self.decomp(x_normalized)
        moving_mean_list = moving_means  # List of trend components

        # Extract the seasonal components
        seasonal = res[:, :, -self.target_num:]  # [B, T, num_target_features]

        # Stack trend components into a tensor for parallel processing
        moving_mean_tensor = torch.stack(moving_mean_list, dim=1)  # Shape: [B, num_trends, T, C]

        B, num_trends, T, C = moving_mean_tensor.shape

        # Reshape for embedding
        moving_mean_tensor = moving_mean_tensor.view(-1, T, C)  # [B * num_trends, T, C]
        x_mark_enc_expanded = x_mark_enc.unsqueeze(1).repeat(1, num_trends, 1, 1).view(-1, T, x_mark_enc.shape[-1])  # [B * num_trends, T, x_mark_enc_dim]

        # Embedding for trends
        enc_out_trend = self.enc_embedding_trends(moving_mean_tensor, x_mark_enc_expanded)  # [B * num_trends, T, D]

        # Process through the model layers
        for layer_idx in range(self.layer):
            enc_out_trend = self.model_trends[layer_idx](enc_out_trend)
            enc_out_trend = self.layer_norm_trends[layer_idx](enc_out_trend)

        # Projection and prediction for trends
        dec_out_trend = self.projection_trends(enc_out_trend)  # [B * num_trends, T, c_in]
        dec_out_trend = self.seq2pred_trends(dec_out_trend.transpose(1, 2)).transpose(1, 2)  # [B * num_trends, pred_len, c_in]

        # Reshape back to [B, num_trends, pred_len, target_num]
        dec_out_trend = dec_out_trend.view(B, num_trends, self.pred_len, -1)[:, :, :, -self.target_num:]

        # Sum over trend components
        dec_out_trends_sum = dec_out_trend.sum(dim=1)  # [B, pred_len, target_num]

        # Process seasonal component directly
        dec_out_seasonal = self.model_seasonal(seasonal)  # Corrected line

        # Combine trends and seasonal components
        dec_out = dec_out_trends_sum + dec_out_seasonal  # [B, pred_len, target_num]

        # De-standardize the combined output
        dec_out = dec_out * stdev + means[:, -self.pred_len:, -self.target_num:]

        # Return the last pred_len steps
        return dec_out  # [B, pred_len, target_num]


