"""EfficientAT MobileNetV3 model vendored from EfficientAT models/mn/model.py.

Source: https://github.com/fschmid56/EfficientAT
License: Apache-2.0

Modifications:
- Imports adapted to relative package imports
- Removed pretrained download logic (handled by scripts/download_pretrained.py)
- Removed helpers.utils dependency (NAME_TO_WIDTH inlined in utils.py)
- Removed print statement from get_model()
"""

from __future__ import annotations

import urllib.parse
from functools import partial
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import torch.nn.functional as F
from torch import Tensor, nn

from .attention_pooling import MultiHeadAttentionPooling
from .inverted_residual import InvertedResidual, InvertedResidualConfig
from .utils import cnn_out_size

# Points to GitHub releases (for reference only; downloading handled separately)
model_url = "https://github.com/fschmid56/EfficientAT/releases/download/v0.0.1/"

pretrained_models = {
    "mn10_as": urllib.parse.urljoin(model_url, "mn10_as_mAP_471.pt"),
    "mn10_im": urllib.parse.urljoin(model_url, "mn10_im.pt"),
    "mn10_im_pytorch": urllib.parse.urljoin(model_url, "mn10_im_pytorch.pt"),
}


class MN(nn.Module):
    """MobileNetV3 adapted for audio classification (from EfficientAT)."""

    def __init__(
        self,
        inverted_residual_setting: List[InvertedResidualConfig],
        last_channel: int,
        num_classes: int = 1000,
        block: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        dropout: float = 0.2,
        in_conv_kernel: int = 3,
        in_conv_stride: int = 2,
        in_channels: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        if not inverted_residual_setting:
            raise ValueError("The inverted_residual_setting should not be empty")
        elif not (
            isinstance(inverted_residual_setting, Sequence)
            and all(isinstance(s, InvertedResidualConfig) for s in inverted_residual_setting)
        ):
            raise TypeError("The inverted_residual_setting should be List[InvertedResidualConfig]")

        if block is None:
            block = InvertedResidual

        from torchvision.ops.misc import ConvNormActivation

        depthwise_norm_layer = norm_layer = (
            norm_layer if norm_layer is not None
            else partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)
        )

        layers: List[nn.Module] = []

        # building first layer
        firstconv_output_channels = inverted_residual_setting[0].input_channels
        layers.append(
            ConvNormActivation(
                in_channels,
                firstconv_output_channels,
                kernel_size=in_conv_kernel,
                stride=in_conv_stride,
                norm_layer=norm_layer,
                activation_layer=nn.Hardswish,
            )
        )

        # get squeeze excitation config
        se_cnf = kwargs.get("se_conf", None)

        # track frequency and time dimensions through the network
        f_dim, t_dim = kwargs.get("input_dims", (128, 1000))
        f_dim = cnn_out_size(f_dim, 1, 1, 3, 2)
        t_dim = cnn_out_size(t_dim, 1, 1, 3, 2)
        for cnf in inverted_residual_setting:
            f_dim = cnf.out_size(f_dim)
            t_dim = cnf.out_size(t_dim)
            cnf.f_dim, cnf.t_dim = f_dim, t_dim
            layers.append(block(cnf, se_cnf, norm_layer, depthwise_norm_layer))

        # building last several layers
        lastconv_input_channels = inverted_residual_setting[-1].out_channels
        lastconv_output_channels = 6 * lastconv_input_channels
        layers.append(
            ConvNormActivation(
                lastconv_input_channels,
                lastconv_output_channels,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=nn.Hardswish,
            )
        )

        self.features = nn.Sequential(*layers)
        self.head_type = kwargs.get("head_type", False)
        if self.head_type == "multihead_attention_pooling":
            self.classifier = MultiHeadAttentionPooling(
                lastconv_output_channels,
                num_classes,
                num_heads=kwargs.get("multihead_attention_heads"),
            )
        elif self.head_type == "fully_convolutional":
            self.classifier = nn.Sequential(
                nn.Conv2d(lastconv_output_channels, num_classes, kernel_size=(1, 1), bias=False),
                nn.BatchNorm2d(num_classes),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
        elif self.head_type == "mlp":
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(start_dim=1),
                nn.Linear(lastconv_output_channels, last_channel),
                nn.Hardswish(inplace=True),
                nn.Dropout(p=dropout, inplace=True),
                nn.Linear(last_channel, num_classes),
            )
        else:
            raise NotImplementedError(
                f"Head '{self.head_type}' unknown. Must be one of: "
                "'mlp', 'fully_convolutional', 'multihead_attention_pooling'"
            )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _forward_impl(
        self, x: Tensor, return_fmaps: bool = False
    ) -> Union[Tuple[Tensor, Tensor], Tuple[Tensor, List[Tensor]]]:
        fmaps = []
        for layer in self.features:
            x = layer(x)
            if return_fmaps:
                fmaps.append(x)

        features = F.adaptive_avg_pool2d(x, (1, 1)).squeeze()
        x = self.classifier(x).squeeze()

        if features.dim() == 1 and x.dim() == 1:
            # squeezed batch dimension
            features = features.unsqueeze(0)
            x = x.unsqueeze(0)

        if return_fmaps:
            return x, fmaps
        else:
            return x, features

    def forward(
        self, x: Tensor
    ) -> Union[Tuple[Tensor, Tensor], Tuple[Tensor, List[Tensor]]]:
        return self._forward_impl(x)


def _mobilenet_v3_conf(
    width_mult: float = 1.0,
    reduced_tail: bool = False,
    dilated: bool = False,
    strides: Tuple[int, ...] = (2, 2, 2, 2),
    **kwargs: Any,
) -> Tuple[List[InvertedResidualConfig], int]:
    reduce_divider = 2 if reduced_tail else 1
    dilation = 2 if dilated else 1

    bneck_conf = partial(InvertedResidualConfig, width_mult=width_mult)
    adjust_channels = partial(InvertedResidualConfig.adjust_channels, width_mult=width_mult)

    inverted_residual_setting = [
        bneck_conf(16, 3, 16, 16, False, "RE", 1, 1),
        bneck_conf(16, 3, 64, 24, False, "RE", strides[0], 1),
        bneck_conf(24, 3, 72, 24, False, "RE", 1, 1),
        bneck_conf(24, 5, 72, 40, True, "RE", strides[1], 1),
        bneck_conf(40, 5, 120, 40, True, "RE", 1, 1),
        bneck_conf(40, 5, 120, 40, True, "RE", 1, 1),
        bneck_conf(40, 3, 240, 80, False, "HS", strides[2], 1),
        bneck_conf(80, 3, 200, 80, False, "HS", 1, 1),
        bneck_conf(80, 3, 184, 80, False, "HS", 1, 1),
        bneck_conf(80, 3, 184, 80, False, "HS", 1, 1),
        bneck_conf(80, 3, 480, 112, True, "HS", 1, 1),
        bneck_conf(112, 3, 672, 112, True, "HS", 1, 1),
        bneck_conf(112, 5, 672, 160 // reduce_divider, True, "HS", strides[3], dilation),
        bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, "HS", 1, dilation),
        bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, "HS", 1, dilation),
    ]
    last_channel = adjust_channels(1280 // reduce_divider)
    return inverted_residual_setting, last_channel


def _mobilenet_v3(
    inverted_residual_setting: List[InvertedResidualConfig],
    last_channel: int,
    **kwargs: Any,
) -> MN:
    return MN(inverted_residual_setting, last_channel, **kwargs)


def get_model(
    num_classes: int = 527,
    pretrained_name: str | None = None,
    width_mult: float = 1.0,
    reduced_tail: bool = False,
    dilated: bool = False,
    strides: Tuple[int, int, int, int] = (2, 2, 2, 2),
    head_type: str = "mlp",
    multihead_attention_heads: int = 4,
    input_dim_f: int = 128,
    input_dim_t: int = 1000,
    se_dims: str = "c",
    se_agg: str = "max",
    se_r: int = 4,
) -> MN:
    """Construct an EfficientAT MobileNetV3 model.

    Args:
        num_classes: Number of output classes.
        pretrained_name: Not used (pretrained loading handled externally).
        width_mult: Width multiplier for channel counts.
        reduced_tail: Scale down network tail.
        dilated: Apply dilated convolution to tail.
        strides: Strides for downsampling stages.
        head_type: Classification head type ('mlp', 'fully_convolutional', 'multihead_attention_pooling').
        multihead_attention_heads: Number of attention heads (if head_type='multihead_attention_pooling').
        input_dim_f: Number of frequency bands (mel bins).
        input_dim_t: Number of time frames.
        se_dims: Squeeze-excitation dimensions ('c', 'f', 't', or combinations).
        se_agg: SE aggregation operation.
        se_r: SE bottleneck ratio.

    Returns:
        MN model instance.
    """
    dim_map = {"c": 1, "f": 2, "t": 3}
    assert len(se_dims) <= 3 and all(s in dim_map for s in se_dims) or se_dims == "none"

    input_dims = (input_dim_f, input_dim_t)
    if se_dims == "none":
        se_dims_list = None
    else:
        se_dims_list = [dim_map[s] for s in se_dims]
    se_conf = dict(se_dims=se_dims_list, se_agg=se_agg, se_r=se_r)

    inverted_residual_setting, last_channel = _mobilenet_v3_conf(
        width_mult=width_mult,
        reduced_tail=reduced_tail,
        dilated=dilated,
        strides=strides,
    )
    return _mobilenet_v3(
        inverted_residual_setting,
        last_channel,
        num_classes=num_classes,
        head_type=head_type,
        multihead_attention_heads=multihead_attention_heads,
        input_dims=input_dims,
        se_conf=se_conf,
    )
