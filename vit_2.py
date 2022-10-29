# 以自底向上的方式来逐步实现ViT模型

# 首先导入相关的依赖库
import torch
import torch.nn.functional as F

from torch import nn
from torch import Tensor
from einops import rearrange
from einops.layers.torch import Rearrange

# -----------------------------------------#
# in_channels = [256, 128, 64, 32]
# patch_size = [4, 8, 16, 32]
# emb_size = [4096, 8192, 16384, 32768]
# depth = [3, 2, 1, 1]
# img_size = [20, 40, 80, 160]
# -----------------------------------------#


# Patches Embeddings的实现
# class PatchEmbedding(nn.Module):
#     def __init__(self, in_channels: int = 256, patch_size: int = 4, emb_size: int = 4096, img_size: int = 20):
#         self.patch_size = patch_size
#         super().__init__()
#         self.projection = nn.Sequential(
#             # 使用一个卷积层而不是一个线性层 -> 性能增加
#             nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),
#             # 将卷积操作后的patch铺平
#             Rearrange('b e (h) (w) -> b (h w) e'),
#         )
#         # 生成一个维度为emb_size的向量当做cls_token
#         # self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
#         # 位置编码信息，一共有(img_size // patch_size)**2 + 1(cls token)个位置向量
#         self.positions = nn.Parameter(torch.randn((img_size // patch_size) ** 2, emb_size))  # 没有使用cls token
#
#     def forward(self, x: Tensor) -> Tensor:
#         # b, _, _, _ = x.shape  # 单独先将batch缓存起来
#         x = self.projection(x)  # 进行卷积和铺平操作
#         # 将cls_token 扩展b次
#         # cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
#         # 将cls token在维度1扩展到输入上
#         # x = torch.cat([cls_tokens, x], dim=1)
#         # 添加位置编码
#         # print(x.shape, self.positions.shape)
#         x += self.positions
#         return x

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 256, patch_size: int = 4, emb_size: int = 4096, img_size: int = 20):
        super().__init__()
        self.patch_size = patch_size
        self.projection = nn.Sequential(
            # 使用一个卷积层而不是一个线性层 -> 性能增加
            nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),
            # 将卷积操作后的patch铺平
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
        # 位置编码信息，一共有(img_size // patch_size)**2 + 1(cls token)个位置向量
        self.positions = nn.Parameter(torch.randn((img_size // patch_size) ** 2, emb_size))  # 没有使用cls token

    def forward(self, x: Tensor) -> Tensor:
        x = self.projection(x)  # 进行卷积和铺平操作
        x += self.positions  # 添加位置编码
        return x


# Attention的实现
# class MultiHeadAttention(nn.Module):
#     def __init__(self, emb_size: int = 4096, num_heads: int = 8, dropout: float = 0):
#         super().__init__()
#         self.emb_size = emb_size
#         self.num_heads = num_heads
#         # 使用单个矩阵一次性计算出queries,keys,values
#         self.qkv = nn.Linear(emb_size, emb_size * 3)
#         self.att_drop = nn.Dropout(dropout)
#         self.projection = nn.Linear(emb_size, emb_size)
#
#     def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
#         # 将queries，keys和values划分为num_heads
#         # print("1qkv's shape: ", self.qkv(x).shape)  # 使用单个矩阵一次性计算出queries,keys,values
#         qkv = rearrange(self.qkv(x), "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads, qkv=3)  # 划分到num_heads个头上
#         # print("2qkv's shape: ", qkv.shape)
#
#         queries, keys, values = qkv[0], qkv[1], qkv[2]
#         # print("queries's shape: ", queries.shape)
#         # print("keys's shape: ", keys.shape)
#         # print("values's shape: ", values.shape)
#
#         # 在最后一个维度上相加
#         energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  # batch, num_heads, query_len, key_len
#         # print("energy's shape: ", energy.shape)
#         if mask is not None:
#             fill_value = torch.finfo(torch.float32).min
#             energy.mask_fill(~mask, fill_value)
#
#         scaling = self.emb_size ** (1 / 2)
#         # print("scaling: ", scaling)
#         att = F.softmax(energy, dim=-1) / scaling
#         # print("att1' shape: ", att.shape)
#         att = self.att_drop(att)
#         # print("att2' shape: ", att.shape)
#
#         # 在第三个维度上相加
#         out = torch.einsum('bhal, bhlv -> bhav ', att, values)
#         # print("out1's shape: ", out.shape)
#         out = rearrange(out, "b h n d -> b n (h d)")
#         # print("out2's shape: ", out.shape)
#         out = self.projection(out)
#         # print("out3's shape: ", out.shape)
#         return out

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int = 4096, num_heads: int = 8, dropout: float = 0):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        # 使用单个矩阵一次性计算出queries,keys,values
        self.qkv = nn.Linear(emb_size, emb_size * 3)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        # 将queries，keys和values划分为num_heads
        qkv = rearrange(self.qkv(x), "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads, qkv=3)  # 划分到num_heads个头上

        queries, keys, values = qkv[0], qkv[1], qkv[2]

        # 在最后一个维度上相加
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  # batch, num_heads, query_len, key_len
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy, dim=-1) / scaling
        att = self.att_drop(att)

        # 在第三个维度上相加
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


# Residuals的实现
class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


# MLP的实现
class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size: int = 4096, expansion: int = 4, drop_p: float = 0):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


# Transformer Encoder Block的实现
class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size: int = 4096,
                 drop_p: float = 0,
                 forward_expansion: int = 4,
                 forward_drop_p: float = 0,
                 ** kwargs):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, **kwargs),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))


# Transformer Encoder实现
class TransformerEncoder(nn.Sequential):
    def __init__(self, depth: int = 3, **kwargs):
        super().__init__(*[TransformerEncoderBlock(**kwargs) for _ in range(depth)])


if __name__ == "__main__":
    x = torch.rand(1, 256, 20, 20)
    print("x的形状：", x.shape)

    patchembedding = PatchEmbedding(256, 4, 4096, 20)
    patches_embedded = patchembedding(x)
    transformer = TransformerEncoder(3)
    out = transformer(patches_embedded).reshape(1, 256, 20, 20)
    print(out.shape)



