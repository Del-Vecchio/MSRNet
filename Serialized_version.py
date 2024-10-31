from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from layers.Embed import DataEmbedding
from layers.MSGBlock import GraphBlock, Attention_Block, Predict
from layers.PDF_backbone import PDF_backbone
from torch import Tensor
import copy

# 第四版， 可以处理多个负荷项

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
        # print("x", x.shape) # x torch.Size([32, 96, 32])
        x_gconv = self.gconv(x)
        # print("x_gconv", x_gconv.shape) # x_gconv torch.Size([32, 96, 32])
        # x_gconv = x
        # Apply Linear layer
        out = self.norm(self.linear0(x_gconv))
        # print("out", out.shape) # out torch.Size([32, 96, 32])
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
        self.model = PDF_backbone(c_in=c_in, context_window=context_window, target_window=target_window,
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
        # num_target_features = 3  # Number of target features (last 3 features)
        # self.num_target_features = num_target_features

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
        self.enc_embedding_trends = nn.ModuleList([
            DataEmbedding(c_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
            for _ in range(self.num_trends)
        ])

        # For seasonal components, use num_target_features as input channels
        self.enc_embedding_seasonal = DataEmbedding(self.target_num, configs.d_model,
                                                    configs.embed, configs.freq, configs.dropout)

        # Models for trends and seasonal components
        self.layer = configs.e_layers

        # Trend components: each has its own model
        self.model_trends = nn.ModuleList([
            nn.ModuleList([
                TrendBlock(configs)
                for _ in range(self.layer)
            ])
            for _ in range(self.num_trends)
        ])

        # Seasonal component: uses attention and scale
        # Update configs for seasonal block
        seasonal_configs = copy.deepcopy(configs)
        seasonal_configs.enc_in = self.target_num
        self.model_seasonal = SeasonalBlock(seasonal_configs)

        # Layer norms
        self.layer_norm_trends = nn.ModuleList([
            nn.ModuleList([
                nn.LayerNorm(configs.d_model)
                for _ in range(self.layer)
            ])
            for _ in range(self.num_trends)
        ])
        self.layer_norm_seasonal = nn.ModuleList([
            nn.LayerNorm(configs.d_model) for _ in range(self.layer)
        ])

        # Projection layers
        self.projection_trends = nn.ModuleList([
            nn.Linear(configs.d_model, c_in, bias=True) for _ in range(self.num_trends)
        ])
        self.projection_seasonal = nn.Linear(configs.d_model, self.target_num, bias=True)
        self.trend = nn.Sequential(
            nn.Linear(configs.seq_len, configs.d_model),
            nn.ReLU(),
            nn.Linear(configs.d_model, configs.d_model),
            nn.ReLU(),
            nn.Linear(configs.d_model, configs.pred_len)
        )
        # Prediction layers
        self.seq2pred_trends = nn.ModuleList([
            Predict(configs.individual, c_in,
                    configs.seq_len, configs.pred_len, configs.dropout)
            for _ in range(self.num_trends)
        ])
        self.seq2pred_seasonal = Predict(configs.individual, self.target_num,
                                         configs.seq_len, configs.pred_len, configs.dropout)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # Decompose x_enc into trends and seasonal components
        moving_means, res = self.decomp(x_enc)  # moving_means: list of [B, T, c_in], res: [B, T, c_in]
        moving_mean_list = moving_means  # List of trend components

        # Extract the seasonal components of the last three features
        seasonal = res[:, :, -self.target_num:]  # [B, T, num_target_features]


        # Recalculate period dynamically
        # period, scale_weight = FFT_for_Period(seasonal.mean(dim=-1), self.k)
        # self.configs.period = period
        # self.configs.scale_weight = scale_weight

        trend_outputs = []
        for idx, moving_mean in enumerate(moving_mean_list):
            # Standardize trend components
            means_trend = moving_mean.mean(dim=1, keepdim=True).detach()
            trends_normalized = moving_mean - means_trend
            stdev_trend = torch.sqrt(torch.var(trends_normalized, dim=1, keepdim=True, unbiased=False) + 1e-5)
            trends_normalized /= stdev_trend

            # Embedding
            enc_out_trend = self.enc_embedding_trends[idx](trends_normalized, x_mark_enc)

            # Process through the model
            for layer_idx in range(self.layer):
                enc_out_trend = self.layer_norm_trends[idx][layer_idx](
                    self.model_trends[idx][layer_idx](enc_out_trend))

            # Project back
            dec_out_trend = self.projection_trends[idx](enc_out_trend)  # [B, T, c_in]
            # dec_out_trend = self.trend(dec_out_trend.permute(0, 2, 1)).permute(0, 2, 1)
            dec_out_trend = self.seq2pred_trends[idx](dec_out_trend.transpose(1, 2)).transpose(1, 2)  # [B, pred_len, c_in]
            # print("dec_out_trend", dec_out_trend.shape) # dec_out_trend torch.Size([32, 96, 6])

            # De-standardize trend components
            dec_out_trend = dec_out_trend * stdev_trend
            dec_out_trend = dec_out_trend + means_trend

            # Extract the trend of the last three features
            dec_out_trend_last_features = dec_out_trend[:, :, -self.target_num:]  # [B, pred_len, num_target_features]

            trend_outputs.append(dec_out_trend_last_features)

        # Sum over the trend components
        dec_out_trends_sum = torch.stack(trend_outputs).sum(dim=0)  # [B, pred_len, num_target_features]

        # Process seasonal component
        dec_out_seasonal = self.model_seasonal(seasonal)

        # Combine trends and seasonal components
        dec_out = dec_out_trends_sum + dec_out_seasonal  # [B, pred_len, num_target_features]

        # Return the last pred_len steps
        return dec_out[:, -self.pred_len:, :]  # [B, pred_len, num_target_features]

# 先进行去均值和方差的归一化，在进行分解，最后季节和趋势项相加后的结果再反归一化
# def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
#     # Standardize the entire input data
#     means = x_enc.mean(dim=1, keepdim=True).detach()
#     x_normalized = x_enc - means
#     stdev = torch.sqrt(torch.var(x_normalized, dim=1, keepdim=True, unbiased=False) + 1e-5)
#     x_normalized /= stdev
#
#     # Decompose the normalized data
#     moving_means, res = self.decomp(x_normalized)
#     moving_mean_list = moving_means  # List of trend components
#
#     # Extract the seasonal components
#     seasonal = res[:, :, -self.target_num:]  # [B, T, num_target_features]
#
#     trend_outputs = []
#     for idx, moving_mean in enumerate(moving_mean_list):
#         # Remove individual standardization of trend components
#         enc_out_trend = self.enc_embedding_trends[idx](moving_mean, x_mark_enc)
#
#         # Process through the model
#         for layer_idx in range(self.layer):
#             enc_out_trend = self.layer_norm_trends[idx][layer_idx](
#                 self.model_trends[idx][layer_idx](enc_out_trend))
#
#         # Project back
#         dec_out_trend = self.projection_trends[idx](enc_out_trend)  # [B, T, c_in]
#         dec_out_trend = self.seq2pred_trends[idx](dec_out_trend.transpose(1, 2)).transpose(1, 2)  # [B, pred_len, c_in]
#
#         trend_outputs.append(dec_out_trend[:, :, -self.target_num:])  # Extract last features
#
#     # Sum over the trend components
#     dec_out_trends_sum = torch.stack(trend_outputs).sum(dim=0)  # [B, pred_len, num_target_features]
#
#     # Process seasonal component
#     dec_out_seasonal = self.model_seasonal(seasonal)
#
#     # Combine trends and seasonal components
#     dec_out = dec_out_trends_sum + dec_out_seasonal  # [B, pred_len, num_target_features]
#
#     # De-standardize the combined output
#     #dec_out = dec_out * stdev + means[:, -self.pred_len:, -self.target_num:]
#     dec_out = dec_out * stdev[:, :, -self.target_num:] + means[:, :, -self.target_num:]
#
#     # Return the last pred_len steps
#     return dec_out  # [B, pred_len, num_target_features]
