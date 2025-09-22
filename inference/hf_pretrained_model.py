import sys

from transformers import PretrainedConfig, PreTrainedModel

sys.path.append('../')
from src.model import VisionTransformer

class TemporalViTConfig(PretrainedConfig):
    model_type = 'temporal-vit'

    def __init__(self,
                 img_wh: int = 224,
                 in_channels: int = 3,
                 patch_size: int = 16,
                 embd_dim: int = 768,
                 patch_embd_bias: bool = True,
                 n_frames: int = 8,
                 num_heads: int = 12,
                 attention_dropout_p: float = 0.0,
                 attn_proj_dropout_p: float = 0.0,
                 use_flash_attention: bool = True,
                 mlp_factor: int = 4,
                 mlp_dropout_p: float = 0.0,
                 head_dropout_p: float = 0.0,
                 num_blocks: int = 12,
                 num_classes: int = 1000,
                 pooling: str = 'cls',
                 custom_init: bool = True,
                 pos_dropout_p: float = 0.0,
                 custom_weight_init: bool = True,
                 **kwargs):
        super().__init__(**kwargs)
        self.img_wh = img_wh
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.embd_dim = embd_dim
        self.patch_embd_bias = patch_embd_bias
        self.n_frames = n_frames
        self.num_heads = num_heads
        self.attention_dropout_p = attention_dropout_p
        self.attn_proj_dropout_p = attn_proj_dropout_p
        self.use_flash_attention = use_flash_attention
        self.mlp_factor = mlp_factor
        self.mlp_dropout_p = mlp_dropout_p
        self.head_dropout_p = head_dropout_p
        self.num_blocks = num_blocks
        self.num_classes = num_classes
        self.pooling = pooling
        self.custom_init = custom_init
        self.pos_dropout_p = pos_dropout_p
        self.custom_weight_init = custom_weight_init


class TemporalViTHF(PreTrainedModel):
    config_class = TemporalViTConfig
    
    def __init__(self, config):
        config.custom_weight_init = False
        super().__init__(config)
        self.model = VisionTransformer(config)

    def forward(self, x):
        return self.model(x)

