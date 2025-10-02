import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class ViTConfig:
    """
    Configuration class for the Vision Transformer (ViT).

    :param img_wh: Width and height of the input image (default: 224).
    :type img_wh: int
    :param in_channels: Number of input channels (default: 3).
    :type in_channels: int
    :param patch_size: Size of each patch (default: 16).
    :type patch_size: int
    :param embd_dim: Embedding dimension (default: 768).
    :type embd_dim: int
    :param patch_embd_bias: Whether to use bias in patch embedding (default: True).
    :type patch_embd_bias: bool
    :param n_frames: Number of frames for temporal modeling (default: 8).
    :type n_frames: int
    :param num_heads: Number of attention heads (default: 12).
    :type num_heads: int
    :param attention_dropout_p: Dropout probability for attention layers (default: 0.0).
    :type attention_dropout_p: float
    :param attn_proj_dropout_p: Dropout probability for attention projection layers (default: 0.0).
    :type attn_proj_dropout_p: float
    :param use_flash_attention: Whether to use flash attention (default: True).
    :type use_flash_attention: bool
    :param mlp_factor: Expansion factor for the MLP layers (default: 4).
    :type mlp_factor: int
    :param mlp_dropout_p: Dropout probability for MLP layers (default: 0.0).
    :type mlp_dropout_p: float
    :param head_dropout_p: Dropout probability for the classification head (default: 0.0).
    :type head_dropout_p: float
    :param num_blocks: Number of encoder blocks (default: 12).
    :type num_blocks: int
    :param num_classes: Number of output classes (default: 1000).
    :type num_classes: int
    :param pooling: Pooling strategy, either 'cls' or 'mean' (default: 'cls').
    :type pooling: str
    :param custom_init: Whether to use custom weight initialization (default: True).
    :type custom_init: bool
    :param pos_dropout_p: Dropout probability for positional embeddings (default: 0.0).
    :type pos_dropout_p: float
    :param custom_weight_init: Whether to use custom weight initialization (default: True).
    :type custom_weight_init: bool
    """
    img_wh: int = 224
    in_channels: int = 3
    patch_size: int = 16
    embd_dim: int = 768
    patch_embd_bias: bool = True
    n_frames: int = 8

    num_heads: int = 12
    attention_dropout_p: float = 0.0
    attn_proj_dropout_p: float = 0.0
    use_flash_attention: bool = True

    mlp_factor: int = 4
    mlp_dropout_p: float = 0.0

    head_dropout_p: float = 0.0
    num_blocks: int = 12
    num_classes: int = 1000
    pooling: str = 'cls'
    custom_init: bool = True
    pos_dropout_p: float = 0.0
    custom_weight_init: bool = True

class PatchEmbedding(nn.Module):
    """
    Patch embedding layer for the Vision Transformer.

    :param config: Configuration object of type :class:`ViTConfig`.
    :type config: ViTConfig
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        assert config.img_wh % config.patch_size == 0
        self.num_patches = config.img_wh // config.patch_size

        self.conv = nn.Conv2d(in_channels=config.in_channels,
                              out_channels=config.embd_dim,
                              kernel_size=config.patch_size,
                              stride=config.patch_size,
                              bias=config.patch_embd_bias)

    def forward(self, x):
        """
        Forward pass for the patch embedding layer.

        :param x: Input tensor of shape (B, T, C, H, W).
        :type x: torch.Tensor
        :return: Embedded patches of shape (B, T*num_patches^2, embd_dim).
        :rtype: torch.Tensor
        """
        B, T, C, H, W = x.shape
        x = x.reshape(B*T, C, H, W)
        x = self.conv(x) # (B*T, E, self.num_patches, self.num_patches)
        x = x.reshape(B, T, self.config.embd_dim, self.num_patches*self.num_patches).transpose(2, 3).reshape(B, T*self.num_patches*self.num_patches, self.config.embd_dim) # (B, self.num_patches^2, E)
        return x

class SelfAttention(nn.Module):
    """
    Self-attention mechanism for the Vision Transformer.

    :param config: Configuration object of type :class:`ViTConfig`.
    :type config: ViTConfig
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        assert config.embd_dim % config.num_heads == 0
        self.head_size = config.embd_dim // config.num_heads
        self.use_flash_attention = config.use_flash_attention
        self.scale = config.embd_dim ** (-0.5)

        self.q = nn.Linear(config.embd_dim, config.embd_dim)
        self.k = nn.Linear(config.embd_dim, config.embd_dim)
        self.v = nn.Linear(config.embd_dim, config.embd_dim)
        self.attn_dropout = nn.Dropout(config.attention_dropout_p)

        self.proj = nn.Linear(config.embd_dim, config.embd_dim)
        self.proj_dropout = nn.Dropout(config.attn_proj_dropout_p)

    def forward(self, x):
        """
        Forward pass for the self-attention layer.

        :param x: Input tensor of shape (B, seq_len, embd_dim).
        :type x: torch.Tensor
        :return: Output tensor of shape (B, seq_len, embd_dim).
        :rtype: torch.Tensor
        """
        batch_size, seq_len, embd_dim = x.shape
        q = self.q(x).reshape(batch_size, seq_len, self.config.num_heads, self.head_size).transpose(1, 2) # (B, num_heads, seq_len, head_size)
        k = self.k(x).reshape(batch_size, seq_len, self.config.num_heads, self.head_size).transpose(1, 2) # (B, num_heads, seq_len, head_size)
        v = self.v(x).reshape(batch_size, seq_len, self.config.num_heads, self.head_size).transpose(1, 2) # (B, num_heads, seq_len, head_size)

        if self.use_flash_attention:
            x = F.scaled_dot_product_attention(q,k,v,dropout_p=self.config.attention_dropout_p if self.training else 0.0)
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = F.softmax(attn, dim=-1)
            attn = self.attn_dropout(attn) # (B, num_heads, seq_len, seq_len)
            x = attn @ v # (B, num_heads, seq_len, head_size)

        x = x.transpose(1, 2).reshape(batch_size, seq_len, embd_dim)

        x = self.proj(x)
        x = self.proj_dropout(x)

        return x # (B, seq_len, E)

class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) for feature transformation.

    :param config: Configuration object of type :class:`ViTConfig`.
    :type config: ViTConfig
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.linear1 = nn.Linear(config.embd_dim, config.mlp_factor*config.embd_dim)
        self.activation_f = nn.GELU()
        self.dropout1 = nn.Dropout(config.mlp_dropout_p)
        self.linear2 = nn.Linear(config.mlp_factor*config.embd_dim, config.embd_dim)
        self.dropout2 = nn.Dropout(config.mlp_dropout_p)

    def forward(self, x):
        """
        Forward pass for the MLP layer.

        :param x: Input tensor of shape (B, seq_len, embd_dim).
        :type x: torch.Tensor
        :return: Transformed tensor of shape (B, seq_len, embd_dim).
        :rtype: torch.Tensor
        """
        x = self.activation_f(self.linear1(x))
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.dropout2(x)

        return x

class EncoderBlock(nn.Module):
    """
    Encoder block for the Vision Transformer.

    :param config: Configuration object of type :class:`ViTConfig`.
    :type config: ViTConfig
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.norm1 = nn.LayerNorm(config.embd_dim)
        self.attn = SelfAttention(config)
        self.norm2 = nn.LayerNorm(config.embd_dim)
        self.mlp = MLP(config)

    def forward(self, x):
        """
        Forward pass for the encoder block.

        :param x: Input tensor of shape (B, seq_len, embd_dim).
        :type x: torch.Tensor
        :return: Output tensor of shape (B, seq_len, embd_dim).
        :rtype: torch.Tensor
        """
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))

        return x

class VisionTransformer(nn.Module):
    """
    Vision Transformer (ViT) model for image classification.

    :param config: Configuration object of type :class:`ViTConfig`.
    :type config: ViTConfig
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.patch_embd = PatchEmbedding(config)

        self.seq_len = self.patch_embd.num_patches ** 2
        self.temporal_seq_len = config.n_frames * self.seq_len

        if self.config.pooling == 'cls':
            self.seq_len += 1
            self.temporal_seq_len += 1

        self.pos_embed = nn.Parameter(torch.zeros(size=(1,self.seq_len,config.embd_dim)))
        self.pos_dropout = nn.Dropout(config.pos_dropout_p)
        self.cls_token = nn.Parameter(torch.zeros(size=(1,1,config.embd_dim)))

        f_position = torch.arange(config.n_frames).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, config.embd_dim, 2) * (-math.log(10000.0)))
        f_pe = torch.zeros(config.n_frames, config.embd_dim)
        f_pe[:, 0::2] = torch.sin(f_position * div_term)
        f_pe[:, 1::2] = torch.cos(f_position * div_term)
        f_pe = f_pe.repeat(self.temporal_seq_len, 1)
        self.register_buffer('f_pe', f_pe, persistent=False)

        position = torch.arange(self.patch_embd.num_patches**2).unsqueeze(1)
        pe = torch.zeros(self.patch_embd.num_patches**2, config.embd_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.repeat(self.temporal_seq_len, 1)
        self.register_buffer('pe', pe, persistent=False)


        self.blocks = nn.ModuleList([EncoderBlock(config) for _ in range(config.num_blocks)])

        self.norm = nn.LayerNorm(config.embd_dim)

        self.head_dropout = nn.Dropout(config.head_dropout_p)
        self.head = nn.Linear(config.embd_dim, config.num_classes)

        if config.custom_weight_init: self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, VisionTransformer):
            module.cls_token.data = nn.init.trunc_normal_(module.cls_token.data, mean=0, std=0.02)
            # module.pos_embed.data = nn.init.trunc_normal_(module.pos_embed.data, mean=0, std=0.02)
        elif isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x):
        """
        Forward pass for the Vision Transformer.

        :param x: Input tensor of shape (B, in_channels, img_h, img_w).
        :type x: torch.Tensor
        :return: Output tensor of shape (B, num_classes).
        :rtype: torch.Tensor
        """
        # x: (B, in_channels, img_h, img_w)
        embds = self.patch_embd(x) #(B, num_patches^2, embd_dim)

        if self.config.pooling == 'cls':
            cls_expand = self.cls_token.expand(x.shape[0], -1, -1)
            embds = torch.cat([embds, cls_expand], dim=1)

        embds = embds + self.pe[:embds.shape[1]] + self.f_pe[:embds.shape[1]]
        # embds = self.pos_dropout(embds)

        x = embds
        for layer in self.blocks:
            x = layer(x)

        x = self.norm(x)

        if self.config.pooling == 'cls':
            x = x[:,-1]
        else:
            x = x.mean(dim=1)

        x = self.head_dropout(x)
        x = self.head(x)

        return x


if __name__ == '__main__':
    rand = torch.rand(size=(4, 8, 3, 224, 224))
    vit_config = ViTConfig()
    vit = VisionTransformer(vit_config)
    print(vit(rand).shape)
    print(vit(rand))
