import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class MaxPool3dSamePadding(nn.MaxPool3d):
    def compute_pad(self, dim, s):
        if s % self.stride[dim] == 0:
            return max(self.kernel_size[dim] - self.stride[dim], 0)
        else:
            return max(self.kernel_size[dim] - (s % self.stride[dim]), 0)

    def forward(self, x):
        (batch, channel, t, h, w) = x.size()
        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)

        pad_t_f = pad_t // 2
        pad_t_b = pad_t - pad_t_f
        pad_h_f = pad_h // 2
        pad_h_b = pad_h - pad_h_f
        pad_w_f = pad_w // 2
        pad_w_b = pad_w - pad_w_f

        pad = (pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b)
        x = F.pad(x, pad)
        return super().forward(x)


class Unit3D(nn.Module):
    def __init__(
        self,
        in_channels,
        output_channels,
        kernel_shape=(1, 1, 1),
        stride=(1, 1, 1),
        padding=0,
        activation_fn=F.relu,
        use_batch_norm=True,
        use_bias=False,
        name="unit_3d",
    ):
        """Initializes Unit3D module."""
        super().__init__()

        self._output_channels = output_channels
        self._kernel_shape = kernel_shape
        self._stride = stride
        self._use_batch_norm = use_batch_norm
        self._activation_fn = activation_fn
        self._use_bias = use_bias
        self.name = name
        self.padding = padding

        self.conv3d = nn.Conv3d(
            in_channels=in_channels,
            out_channels=self._output_channels,
            kernel_size=self._kernel_shape,
            stride=self._stride,
            padding=0,  # we always want padding to be 0 here. We will dynamically pad based on input size in forward function
            bias=self._use_bias,
        )

        if self._use_batch_norm:
            self.bn = nn.BatchNorm3d(self._output_channels, eps=0.001, momentum=0.01)

    def compute_pad(self, dim, s):
        if s % self._stride[dim] == 0:
            return max(self._kernel_shape[dim] - self._stride[dim], 0)
        else:
            return max(self._kernel_shape[dim] - (s % self._stride[dim]), 0)

    def forward(self, x):
        (batch, channel, t, h, w) = x.size()
        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)

        pad_t_f = pad_t // 2
        pad_t_b = pad_t - pad_t_f
        pad_h_f = pad_h // 2
        pad_h_b = pad_h - pad_h_f
        pad_w_f = pad_w // 2
        pad_w_b = pad_w - pad_w_f

        pad = (pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b)
        x = F.pad(x, pad)

        x = self.conv3d(x)
        if self._use_batch_norm:
            x = self.bn(x)
        if self._activation_fn is not None:
            x = self._activation_fn(x)
        return x


class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels, name):
        super().__init__()

        self.b0 = Unit3D(
            in_channels=in_channels,
            output_channels=out_channels[0],
            kernel_shape=[1, 1, 1],
            padding=0,
            name=name + "/Branch_0/Conv3d_0a_1x1",
        )
        self.b1a = Unit3D(
            in_channels=in_channels,
            output_channels=out_channels[1],
            kernel_shape=[1, 1, 1],
            padding=0,
            name=name + "/Branch_1/Conv3d_0a_1x1",
        )
        self.b1b = Unit3D(
            in_channels=out_channels[1],
            output_channels=out_channels[2],
            kernel_shape=[3, 3, 3],
            name=name + "/Branch_1/Conv3d_0b_3x3",
        )
        self.b2a = Unit3D(
            in_channels=in_channels,
            output_channels=out_channels[3],
            kernel_shape=[1, 1, 1],
            padding=0,
            name=name + "/Branch_2/Conv3d_0a_1x1",
        )
        self.b2b = Unit3D(
            in_channels=out_channels[3],
            output_channels=out_channels[4],
            kernel_shape=[3, 3, 3],
            name=name + "/Branch_2/Conv3d_0b_3x3",
        )
        self.b3a = MaxPool3dSamePadding(kernel_size=[3, 3, 3], stride=(1, 1, 1), padding=0)
        self.b3b = Unit3D(
            in_channels=in_channels,
            output_channels=out_channels[5],
            kernel_shape=[1, 1, 1],
            padding=0,
            name=name + "/Branch_3/Conv3d_0b_1x1",
        )
        self.name = name

    def forward(self, x):
        b0 = self.b0(x)
        b1 = self.b1b(self.b1a(x))
        b2 = self.b2b(self.b2a(x))
        b3 = self.b3b(self.b3a(x))
        return torch.cat([b0, b1, b2, b3], dim=1)


class InceptionI3d(nn.Module):
    """Inception-v1 I3D architecture.
    The model is introduced in:
        Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset
        Joao Carreira, Andrew Zisserman
        https://arxiv.org/pdf/1705.07750v1.pdf.
    See also the Inception architecture, introduced in:
        Going deeper with convolutions
        Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed,
        Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich.
        http://arxiv.org/pdf/1409.4842v1.pdf.
    """

    # Endpoints of the model in order. During construction, all the endpoints up
    # to a designated `final_endpoint` are returned in a dictionary as the
    # second return value.
    VALID_ENDPOINTS = (
        "Conv3d_1a_7x7",
        "MaxPool3d_2a_3x3",
        "Conv3d_2b_1x1",
        "Conv3d_2c_3x3",
        "MaxPool3d_3a_3x3",
        "Mixed_3b",
        "Mixed_3c",
        "MaxPool3d_4a_3x3",
        "Mixed_4b",
        "Mixed_4c",
        "Mixed_4d",
        "Mixed_4e",
        "Mixed_4f",
        "MaxPool3d_5a_2x2",
        "Mixed_5b",
        "Mixed_5c",
        "Logits",
        "Predictions",
    )

    def __init__(
        self,
        time_spatial_squeeze=True,
        final_endpoint="Logits",
        name="inception_i3d",
        in_channels=3,
    ):
        """Initializes I3D model instance.
        Args:
          num_classes: The number of outputs in the logit layer (default 400, which
              matches the Kinetics dataset).
          spatial_squeeze: Whether to squeeze the spatial dimensions for the logits
              before returning (default True).
          final_endpoint: The model contains many possible endpoints.
              `final_endpoint` specifies the last endpoint for the model to be built
              up to. In addition to the output at `final_endpoint`, all the outputs
              at endpoints up to `final_endpoint` will also be returned, in a
              dictionary. `final_endpoint` must be one of
              InceptionI3d.VALID_ENDPOINTS (default 'Logits').
          name: A string (optional). The name of this module.
        Raises:
          ValueError: if `final_endpoint` is not recognized.
        """

        if final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError(f"Unknown final endpoint {final_endpoint}")

        super().__init__()
        self._time_spatial_squeeze = time_spatial_squeeze
        self._final_endpoint = final_endpoint
        self.logits = None

        if self._final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError(f"Unknown final endpoint {self._final_endpoint}")

        self.end_points = {}
        end_point = "Conv3d_1a_7x7"
        self.end_points[end_point] = Unit3D(
            in_channels=in_channels,
            output_channels=64,
            kernel_shape=[7, 7, 7],
            stride=(2, 2, 2),
            padding=(3, 3, 3),
            name=name + end_point,
        )
        if self._final_endpoint == end_point:
            return

        end_point = "MaxPool3d_2a_3x3"
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[1, 3, 3], stride=(1, 2, 2), padding=0)
        if self._final_endpoint == end_point:
            return

        end_point = "Conv3d_2b_1x1"
        self.end_points[end_point] = Unit3D(
            in_channels=64,
            output_channels=64,
            kernel_shape=[1, 1, 1],
            padding=0,
            name=name + end_point,
        )
        if self._final_endpoint == end_point:
            return

        end_point = "Conv3d_2c_3x3"
        self.end_points[end_point] = Unit3D(
            in_channels=64,
            output_channels=192,
            kernel_shape=[3, 3, 3],
            padding=1,
            name=name + end_point,
        )
        if self._final_endpoint == end_point:
            return

        end_point = "MaxPool3d_3a_3x3"
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[1, 3, 3], stride=(1, 2, 2), padding=0)
        if self._final_endpoint == end_point:
            return

        end_point = "Mixed_3b"
        self.end_points[end_point] = InceptionModule(192, [64, 96, 128, 16, 32, 32], name + end_point)
        if self._final_endpoint == end_point:
            return

        end_point = "Mixed_3c"
        self.end_points[end_point] = InceptionModule(256, [128, 128, 192, 32, 96, 64], name + end_point)
        if self._final_endpoint == end_point:
            return

        end_point = "MaxPool3d_4a_3x3"
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[3, 3, 3], stride=(2, 2, 2), padding=0)
        if self._final_endpoint == end_point:
            return

        end_point = "Mixed_4b"
        self.end_points[end_point] = InceptionModule(128 + 192 + 96 + 64, [192, 96, 208, 16, 48, 64], name + end_point)
        if self._final_endpoint == end_point:
            return

        end_point = "Mixed_4c"
        self.end_points[end_point] = InceptionModule(192 + 208 + 48 + 64, [160, 112, 224, 24, 64, 64], name + end_point)
        if self._final_endpoint == end_point:
            return

        end_point = "Mixed_4d"
        self.end_points[end_point] = InceptionModule(160 + 224 + 64 + 64, [128, 128, 256, 24, 64, 64], name + end_point)
        if self._final_endpoint == end_point:
            return

        end_point = "Mixed_4e"
        self.end_points[end_point] = InceptionModule(128 + 256 + 64 + 64, [112, 144, 288, 32, 64, 64], name + end_point)
        if self._final_endpoint == end_point:
            return

        end_point = "Mixed_4f"
        self.end_points[end_point] = InceptionModule(
            112 + 288 + 64 + 64, [256, 160, 320, 32, 128, 128], name + end_point
        )
        if self._final_endpoint == end_point:
            return

        end_point = "MaxPool3d_5a_2x2"
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[1, 2, 2], stride=(1, 2, 2), padding=0)
        if self._final_endpoint == end_point:
            return

        end_point = "Mixed_5b"
        self.end_points[end_point] = InceptionModule(
            256 + 320 + 128 + 128, [256, 160, 320, 32, 128, 128], name + end_point
        )
        if self._final_endpoint == end_point:
            return

        end_point = "Mixed_5c"
        self.end_points[end_point] = InceptionModule(
            256 + 320 + 128 + 128, [384, 192, 384, 48, 128, 128], name + end_point
        )

        if self._final_endpoint == end_point:
            return

        self.build()

    def build(self):
        for k in self.end_points.keys():
            self.add_module(k, self.end_points[k])

    def get_out_size(self, shape: Sequence[int], dim=None) -> int:
        device = next(self.parameters()).device
        out = self(torch.zeros((1, *shape), device=device))
        return out.size(dim)

    def forward(self, x):
        for end_point in self.VALID_ENDPOINTS:
            if end_point in self.end_points:
                x = self._modules[end_point](x)  # use _modules to work with dataparallel
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, : x.size(1), :]
        return x


class CrossAttention(nn.Module):
    def __init__(self, dim_q, dim_k, dim_v, dim_out, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim_out // num_heads
        assert dim_out % num_heads == 0, "dim_out must be divisible by num_heads"
        self.scale = self.head_dim**-0.5

        self.query_proj = nn.Linear(dim_q, dim_out)
        self.key_proj = nn.Linear(dim_k, dim_out)
        self.value_proj = nn.Linear(dim_v, dim_out)

        self.out_proj = nn.Linear(dim_out, dim_out)

    def forward(self, query, key, value):
        # Linear transformation of query, key, and value
        q = self.query_proj(query)  # shape: (batch_size, query_len, dim_out)
        k = self.key_proj(key)  # shape: (batch_size, key_len, dim_out)
        v = self.value_proj(value)  # shape: (batch_size, value_len, dim_out)

        # Split dimensions for multi-head attention, and compute per head
        # print("q:", q.size(), "k:", k.size(), "v:", v.size())
        q = q.view(q.size(0), q.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(k.size(0), k.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(v.size(0), v.size(1), self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = attn_weights.softmax(dim=-1)

        # Multiply attention weights with values
        attn_output = torch.matmul(attn_weights, v)

        # Concatenate results and return to original dimensions
        attn_output = attn_output.transpose(1, 2).reshape(v.size(0), -1, self.num_heads * self.head_dim)
        output = self.out_proj(attn_output)

        return output, attn_weights


class FeedForward(nn.Module):
    def __init__(self, d_model, hidden, drop_prob=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class PreGRULayer(nn.Module):
    def __init__(
        self,
        d_model,
        num_heads,
        ffn_hidden,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.pre_norm0 = nn.LayerNorm(d_model)
        self.self_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout0 = nn.Dropout(dropout)

        self.pre_norm1 = nn.LayerNorm(d_model)
        self.cross_attention = CrossAttention(
            dim_q=d_model,
            dim_k=d_model,
            dim_v=d_model,
            dim_out=d_model,
            num_heads=num_heads,
        )
        self.dropout1 = nn.Dropout(dropout)

        self.pre_norm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, ffn_hidden)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, q, x) -> torch.Tensor:
        """
        Expected shapes:
            - q: (b, 1, dim_q)
            - x: (b, seq, dim_kv)
        Output shape:
            (b, seq, d_model)
        """

        # cross attention
        _x = x
        x = self.pre_norm1(x)
        x, _ = self.cross_attention(query=q, key=x, value=x)
        x = self.dropout1(x)
        x = x + _x

        # self attention
        _x = x
        x = self.pre_norm0(x)
        x, _ = self.self_attention(query=x, key=x, value=x)
        x = self.dropout0(x)
        x = x + _x

        # pairwise feed foward
        _x = x
        x = self.pre_norm2(x)
        x = self.ffn(x)
        x = self.dropout2(x)
        x = x + _x

        return x


class VariableLengthWaypointPredictor(nn.Module):
    """Variable-length GRU-based waypoint predictor with optional timestamp inputs."""

    def __init__(
        self,
        d_model,
        memory_seq_len,
        timestamp_dim=0,
        waypoint_dim=2,
        num_heads=4,
        start_from_origin=True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.waypoint_dim = waypoint_dim
        self.start_from_origin = start_from_origin

        self.hidden_state = nn.Parameter(torch.randn(1, d_model))
        self.pos_embedding = nn.Parameter(torch.randn(1, memory_seq_len, d_model))

        self.pre_gru_layer = PreGRULayer(
            d_model=d_model,
            num_heads=num_heads,
            ffn_hidden=d_model // 2,
        )
        self.gru = nn.GRUCell(
            input_size=waypoint_dim + d_model + timestamp_dim,
            hidden_size=d_model,
        )
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(d_model // 2, waypoint_dim),  # wp_dim
        )

    def forward(
        self,
        memory: Tensor,  # (b, t, c)
        num_waypoints: int,
        timestamps: Tensor = None,
    ) -> dict[str, Tensor]:
        batch_size = memory.shape[0]
        dtype = memory.dtype

        wp = memory.new_zeros((batch_size, self.waypoint_dim))
        h = self.hidden_state.repeat(batch_size, 1).to(dtype)
        pos_embedding = self.pos_embedding.repeat(batch_size, 1, 1).to(dtype)
        memory = memory + pos_embedding

        waypoints = []
        if self.start_from_origin:
            # add first waypoint as zero origin
            waypoints.append(memory.new_zeros((batch_size, self.waypoint_dim)))
            num_waypoints = num_waypoints - 1

        for t in range(num_waypoints):
            inputs = self.pre_gru_layer(q=h.unsqueeze(1), x=memory)  # (b, t, c)
            inputs = inputs.mean(1)  # (b, c)
            inputs = torch.cat([wp, inputs], dim=1)

            if timestamps is not None:
                inputs = torch.cat([inputs, timestamps[:, t].reshape(batch_size, -1)], dim=1)

            h = self.gru(inputs, h)
            dx = self.head(h)
            wp = wp + dx
            waypoints.append(wp)

        waypoints = torch.stack(waypoints, dim=1)  # (b, n_wps, wp_dim)

        return waypoints


class VideoActionEstimator(nn.Module):
    def __init__(
        self,
        input_shape,
        num_classes,
        max_seq_len=44,
        timestamp_dim=0,
        d_model=512,
        num_heads=8,
        dropout=0.1,
        feature_map_size=4,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.timestamp_dim = timestamp_dim
        assert input_shape[1] == max_seq_len

        self.backbone = InceptionI3d()
        feature_dim, seq_len = self.backbone.get_out_size(input_shape)[1:3]

        self.avg_pool = nn.AdaptiveAvgPool3d((None, feature_map_size, feature_map_size))
        memory_seq_len = seq_len * feature_map_size**2

        self.squeeze_linear = nn.Linear(feature_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model=d_model, max_len=memory_seq_len)
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=512,
            batch_first=True,
            activation=F.gelu,
        )
        self.self_attn = TransformerEncoder(
            encoder_layer,
            num_layers=2,
        )

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Dropout(p=dropout),
            nn.GELU(),
            nn.Linear(d_model, num_classes),
        )
        self.visual_odmetry = VariableLengthWaypointPredictor(
            d_model=d_model,
            memory_seq_len=memory_seq_len,
            waypoint_dim=2,  # x, y axes
            timestamp_dim=timestamp_dim,
            num_heads=num_heads,
        )

    def forward(self, frames: Tensor, timestamps: Tensor = None) -> dict[str, Tensor]:
        x = frames
        num_frames = x.size(2)  # seq which must be consistent in a batch
        assert (
            num_frames <= self.max_seq_len
        ), f"Input tensor has exceeded sequence length(={num_frames}) than max_seq_len(={self.max_seq_len})"

        x = self.backbone(x)  # (b, 1024, 11, 7, 7)
        x = self.avg_pool(x)  # (b, 1024, 11, 4, 4)

        b, c, t, h, w = x.size()
        x = x.view(b, t * h * w, c)  # (b, 176, 1024)
        x = self.squeeze_linear(x)  # (b, 176, 512)
        x = self.positional_encoding(x)

        x = self.self_attn(x)  # (b, 176, 512)
        latent_tensor = x.mean(1)  # (b, 512)
        logits = self.classifier(latent_tensor)
        waypoints = self.visual_odmetry(x, num_frames, timestamps=timestamps)

        return {
            "command": logits,
            "waypoints": waypoints,
        }
