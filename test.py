# 导入需要的module
import torch
import torch.nn.functional as F

from torch import nn
from torch import Tensor
from einops import rearrange
from einops.layers.torch import Rearrange

# nework_blocks.py中定义的全部module，复制过来了
# 定义SiLU激活函数
class SiLU(nn.Module):
    """export-friendly version of nn.SiLU()"""

    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


def get_activation(name="silu", inplace=True):
    if name == "silu":
        module = nn.SiLU(inplace=inplace)
    elif name == "relu":
        module = nn.ReLU(inplace=inplace)
    elif name == "lrelu":
        module = nn.LeakyReLU(0.1, inplace=inplace)
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module


# 定义基础卷积
class BaseConv(nn.Module):
    """A Conv2d -> Batchnorm -> silu/leaky relu block"""

    def __init__(
        self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="silu"
    ):
        super().__init__()
        # same padding
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


# 定义深度可分离卷积
class DWConv(nn.Module):
    """Depthwise Conv + Conv"""

    def __init__(self, in_channels, out_channels, ksize, stride=1, act="silu"):
        super().__init__()
        self.dconv = BaseConv(
            in_channels,
            in_channels,
            ksize=ksize,
            stride=stride,
            groups=in_channels,
            act=act,
        )
        self.pconv = BaseConv(
            in_channels, out_channels, ksize=1, stride=1, groups=1, act=act
        )

    def forward(self, x):
        x = self.dconv(x)
        return self.pconv(x)


# ------------------------------#
#    残差结构的构建，小的残差结构
# ------------------------------#
class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(
        self,
        in_channels,
        out_channels,
        shortcut=True,
        expansion=0.5,
        depthwise=False,
        act="silu",
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        Conv = DWConv if depthwise else BaseConv
        # --------------------------------------------#
        #  利用1x1卷积进行通道数的缩减。缩减率一般是50%
        # --------------------------------------------#
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        # --------------------------------------------#
        #  利用3x3卷积进行通道数的拓张。并且完成特征提取
        # --------------------------------------------#
        self.conv2 = Conv(hidden_channels, out_channels, 3, stride=1, act=act)
        self.use_add = shortcut and in_channels == out_channels

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        if self.use_add:
            y = y + x
        return y


class ResLayer(nn.Module):
    "Residual layer with `in_channels` inputs."

    def __init__(self, in_channels: int):
        super().__init__()
        mid_channels = in_channels // 2
        self.layer1 = BaseConv(
            in_channels, mid_channels, ksize=1, stride=1, act="lrelu"
        )
        self.layer2 = BaseConv(
            mid_channels, in_channels, ksize=3, stride=1, act="lrelu"
        )

    def forward(self, x):
        out = self.layer2(self.layer1(x))
        return x + out


# 定义SPP结构
class SPPBottleneck(nn.Module):
    """
    Spatial pyramid pooling layer used in YOLOv3-SPP
    """

    def __init__(
        self, in_channels, out_channels, kernel_sizes=(5, 9, 13), activation="silu"
    ):
        super().__init__()
        hidden_channels = in_channels // 2
        # ----------------------------------------#
        #  利用1x1的卷积对通道数进行缩减
        # ----------------------------------------#
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=activation)
        # ----------------------------------------#
        #  利用一个循环构建不同池化核的最大池化
        # ----------------------------------------#
        self.m = nn.ModuleList(
            [
                nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2)
                for ks in kernel_sizes
            ]
        )
        # ----------------------------------------#
        #  堆叠完成之后再利用一个卷积进行通道数的调整
        # ----------------------------------------#
        conv2_channels = hidden_channels * (len(kernel_sizes) + 1)
        self.conv2 = BaseConv(conv2_channels, out_channels, 1, stride=1, act=activation)

    def forward(self, x):
        # ----------------------------------------#
        #  利用1x1的卷积对通道数进行缩减
        # ----------------------------------------#
        x = self.conv1(x)
        # -------------------------------------------------#
        #  对第一个卷积后的结果利用不同池化核的最大池化进行特征提取，
        #  然后和第一个卷积后的结果在通道维进行堆叠
        # -------------------------------------------------#
        x = torch.cat([x] + [m(x) for m in self.m], dim=1)
        # ----------------------------------------#
        #  堆叠完成之后再利用一个卷积进行通道数的调整
        # ----------------------------------------#
        x = self.conv2(x)
        return x


# 定义CSPLayer结构
class CSPLayer(nn.Module):
    """
    C3 in yolov5, CSP Bottleneck with 3 convolutions
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        n=1,
        shortcut=True,
        expansion=0.5,
        depthwise=False,
        act="silu",
    ):
        """
        Args:
            in_channels (int): input channels.
            out_channels (int): output channels.
            n (int): number of Bottlenecks. Default value: 1.
        """
        # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        hidden_channels = int(out_channels * expansion)  # hidden channels
        # --------------------#
        #  主干部分的初次卷积
        # --------------------#
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        # -----------------------#
        #  大的残差边部分的初次卷积
        # -----------------------#
        self.conv2 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        # ----------------------------#
        #  对堆叠的结果进行卷积的处理
        # ----------------------------#
        self.conv3 = BaseConv(2 * hidden_channels, out_channels, 1, stride=1, act=act)
        # ----------------------------------------#
        #  根据循环的次数构建上述Bottleneck残差结构
        # ----------------------------------------#
        module_list = [
            Bottleneck(
                hidden_channels, hidden_channels, shortcut, 1.0, depthwise, act=act
            )
            for _ in range(n)
        ]
        self.m = nn.Sequential(*module_list)
        # self.m = TransformerEncoder(hidden_channels, hidden_channels, 8, n)

    def forward(self, x):
        # ------------------#
        #  x_1是主干部分
        # ------------------#
        x_1 = self.conv1(x)
        # ------------------#
        #  x_2是大的残差边部分
        # ------------------#
        x_2 = self.conv2(x)
        # ---------------------------------------#
        #  主干部分利用残差结构的堆叠继续进行特征提取
        # ---------------------------------------#
        x_1 = self.m(x_1)
        # ---------------------------------------#
        #  主干部分和大的残差边部分在通道维进行堆叠
        # ---------------------------------------#
        x = torch.cat((x_1, x_2), dim=1)
        # ---------------------------------------#
        #  对堆叠的结果进行卷积的处理
        # ---------------------------------------#
        return self.conv3(x)


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size: int, expansion: int = 4, drop_p: float = 0.5):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int, num_heads: int, dropout: float = 0.5):
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


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int, patch_size: int, emb_size: int, img_size: int):
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


class TransformerEncoderBlock(nn.Module):
    def __init__(self, c, num_heads):
        super().__init__()
        # self.c_emb_size = {"256": 4096,
        #                    "128": 8192,
        #                    "64": 16384,
        #                    "32": 32768}
        if c == 256:
            emb_size = 4096
        elif c == 128:
            emb_size = 8192
        elif c == 64:
            emb_size = 16384
        else:
            emb_size = 32768

        self.layernorm = nn.LayerNorm(emb_size)
        self.attention = MultiHeadAttention(emb_size, num_heads)
        self.dropout = nn.Dropout(0.5)
        self.ffn = FeedForwardBlock(emb_size)

    def forward(self, x):
        res1 = x
        x = self.layernorm(x)
        x = self.attention(x)
        x = self.dropout(x)
        x += res1
        res2 = x
        x = self.layernorm(x)
        x = self.ffn(x)
        x = self.dropout(x)
        x += res2
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, c1, c2, num_heads, num_blocks):
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = BaseConv(c1, c2)
        self.trans = nn.Sequential(*[TransformerEncoderBlock(c2, num_heads) for _ in range(num_blocks)])
        self.c2 = c2

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        b, c, h, w = x.shape
        patches_embedded = PatchEmbedding(c, h // 5, (h // 5) ** 2 * c, h)
        x = patches_embedded(x)

        return self.trans(x).unsqueeze(3).reshape(1, self.c2, w, h)


class C3TR(nn.Module):
    """
    CSP Bottleneck with Transformer Encoder
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        n=1,
        expansion=0.5,
        act="silu",
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)  # hidden channels
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv3 = BaseConv(2 * hidden_channels, out_channels, 1, stride=1, act=act)
        self.m = TransformerEncoder(hidden_channels, hidden_channels, 4, n)

    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x_1 = self.m(x_1)
        print("trans输出的x_1:", x_1.shape)
        x = torch.cat((x_1, x_2), dim=1)
        return self.conv3(x)

# 定义Focus网络结构
class Focus(nn.Module):
    """
    Focus width and height information into channel space.（将空间信息转换到通道信息）
    这个模块位于主干网络backbone的一开始，紧接着输入层，称为stem（茎）
    """

    def __init__(self, in_channels, out_channels, ksize=1, stride=1, act="silu"):
        super().__init__()
        self.conv = BaseConv(in_channels * 4, out_channels, ksize, stride, act=act)

    def forward(self, x):
        # shape of x (b,c,w,h) -> y (b,4c,w/2,h/2)
        # -------------------------------------------------#
        # 前两维度不变，后面两个维度取偶数下标对应的值
        # -------------------------------------------------#
        patch_top_left = x[..., ::2, ::2]
        # -----------------------------------------------------------------#
        # 表示前两维度不变，w维度取第0,2,4,6..行的值，而h维度取第1,3,5,7,...列的值
        # -----------------------------------------------------------------#
        patch_top_right = x[..., ::2, 1::2]
        # -----------------------------------------------------------------#
        # 表示前两维度不变，w维度取第1,3,5,7,...行的值，而h维度取第0,2,4,6..列的值
        # -----------------------------------------------------------------#
        patch_bot_left = x[..., 1::2, ::2]
        # -------------------------------------------------------------------#
        # 表示前两维度不变，w维度取第1,3,5,7,...行的值，而h维度取第1,3,5,7,...列的值
        # -------------------------------------------------------------------#
        patch_bot_right = x[..., 1::2, 1::2]
        x = torch.cat(
            (
                patch_top_left,
                patch_bot_left,
                patch_top_right,
                patch_bot_right,
            ),
            dim=1,
        )
        return self.conv(x)


# darknet.py中定义的darknet53
class Darknet(nn.Module):
    # number of blocks from dark2 to dark5.
    depth2blocks = {21: [1, 2, 2, 1], 53: [2, 8, 8, 4]}

    def __init__(
        self,
        depth,
        in_channels=3,
        stem_out_channels=32,
        out_features=("dark2", "dark3", "dark4", "dark5"),  # 修改：增加了dark2
    ):
        """
        Args:
            depth (int): depth of darknet used in model, usually use [21, 53] for this param.
            in_channels (int): number of input channels, for example, use 3 for RGB image.
            stem_out_channels (int): number of output chanels of darknet stem.
                It decides channels of darknet layer2 to layer5.
            out_features (Tuple[str]): desired output layer name.
        """
        super().__init__()
        assert out_features, "please provide output features of Darknet"
        self.out_features = out_features
        self.stem = nn.Sequential(
            BaseConv(in_channels, stem_out_channels, ksize=3, stride=1, act="lrelu"),  # darknet53的第一个cbl
            *self.make_group_layer(stem_out_channels, num_blocks=1, stride=2),  # cbl + Res1
        )
        in_channels = stem_out_channels * 2  # 64

        num_blocks = Darknet.depth2blocks[depth]
        # create darknet with `stem_out_channels` and `num_blocks` layers.
        # to make model structure more clear, we don't use `for` statement in python.
        self.dark2 = nn.Sequential(
            *self.make_group_layer(in_channels, num_blocks[0], stride=2)
        )
        in_channels *= 2  # 128
        self.dark3 = nn.Sequential(
            *self.make_group_layer(in_channels, num_blocks[1], stride=2)
        )
        in_channels *= 2  # 256
        self.dark4 = nn.Sequential(
            *self.make_group_layer(in_channels, num_blocks[2], stride=2)
        )
        in_channels *= 2  # 512

        self.dark5 = nn.Sequential(
            # 1 512 40 40 -> 1 1024 20 20
            *self.make_group_layer(in_channels, num_blocks[3], stride=2),
            # 1024 20 20 -> 1024 20 20 -> 512 20 20 -> 512 20 20
            *self.make_spp_block([in_channels, in_channels * 2], in_channels * 2),
        )

    def make_group_layer(self, in_channels: int, num_blocks: int, stride: int = 1):
        "starts with conv layer then has `num_blocks` `ResLayer`"
        return [
            BaseConv(in_channels, in_channels * 2, ksize=3, stride=stride, act="lrelu"),
            *[(ResLayer(in_channels * 2)) for _ in range(num_blocks)],
        ]

    def make_spp_block(self, filters_list, in_filters):
        m = nn.Sequential(
            *[
                # 2*CBL
                BaseConv(in_filters, filters_list[0], 1, stride=1, act="lrelu"),
                BaseConv(filters_list[0], filters_list[1], 3, stride=1, act="lrelu"),
                # spp
                SPPBottleneck(
                    in_channels=filters_list[1],
                    out_channels=filters_list[0],
                    activation="lrelu",
                ),
                # 2*CBL
                BaseConv(filters_list[0], filters_list[1], 3, stride=1, act="lrelu"),
                BaseConv(filters_list[1], filters_list[0], 1, stride=1, act="lrelu"),
                # 添加1*trans.
                # C3TR(filters_list[0],
                #      filters_list[0],
                #      n=1,
                #      expansion=0.5,
                #      act='silu'
                #      )
            ]

        )
        return m

    def forward(self, x):
        outputs = {}
        print("stem前的x形状：", x.shape)
        x = self.stem(x)
        print("stem后的x形状：", x.shape)
        outputs["stem"] = x
        print("dark2前的x形状：", x.shape)
        x = self.dark2(x)
        print("dark2后的x形状：", x.shape)
        outputs["dark2"] = x
        print("dark3前的x形状：", x.shape)
        x = self.dark3(x)
        print("dark3后的x形状：", x.shape)
        outputs["dark3"] = x
        print("dark4前的x形状：", x.shape)
        x = self.dark4(x)
        print("dark4后的x形状：", x.shape)
        outputs["dark4"] = x
        print("dark5前的x形状：", x.shape)
        x = self.dark5(x)
        print("dark5后的x形状：", x.shape)
        outputs["dark5"] = x
        print("所有的网络名称与对应的输出：", outputs)
        return {k: v for k, v in outputs.items() if k in self.out_features}


# yolo_fpn.py中定义的FPN
class YOLOFPN(nn.Module):
    """
    YOLOFPN module. Darknet 53 is the default backbone of this model.
    """

    def __init__(
        self,
        depth=53,
        in_features=["dark2", "dark3", "dark4", "dark5"],  # 修改：增加了dark2
    ):
        super().__init__()

        self.backbone = Darknet(depth)
        self.in_features = in_features

        # out 1
        self.out1_cbl = self._make_cbl(512, 256, 1)
        print('111111')
        self.out1 = self._make_embedding_([256, 512], 512 + 256)

        # out 2
        self.out2_cbl = self._make_cbl(256, 128, 1)
        self.out2 = self._make_embedding_([128, 256], 256 + 128)

        # -----------------------------------------------------#
        # out 3 新增加的
        self.out3_cbl = self._make_cbl(128, 64, 1)
        self.out3 = self._make_embedding([64, 128], 128 + 64)
        # -----------------------------------------------------#

        # upsample
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

    def _make_cbl(self, _in, _out, ks):
        return BaseConv(_in, _out, ks, stride=1, act="lrelu")

    def _make_embedding_(self, filters_list, in_filters):
        m = nn.Sequential(
            *[
                # 5*CBL
                self._make_cbl(in_filters, filters_list[0], 1),
                self._make_cbl(filters_list[0], filters_list[1], 3),
                self._make_cbl(filters_list[1], filters_list[0], 1),
                self._make_cbl(filters_list[0], filters_list[1], 3),
                self._make_cbl(filters_list[1], filters_list[0], 1),
                # 添加了1*trans.
                C3TR(filters_list[0],
                     filters_list[0],
                     n=1,
                     expansion=0.5,
                     act='silu',
                     )
            ]
        )
        return m

    def _make_embedding(self, filters_list, in_filters):
        m = nn.Sequential(
            *[
                # 5*CBL
                self._make_cbl(in_filters, filters_list[0], 1),
                self._make_cbl(filters_list[0], filters_list[1], 3),
                self._make_cbl(filters_list[1], filters_list[0], 1),
                self._make_cbl(filters_list[0], filters_list[1], 3),
                self._make_cbl(filters_list[1], filters_list[0], 1),
                # 添加了1*trans.
                # C3TR(filters_list[0],
                #      filters_list[0],
                #      n=1,
                #      expansion=0.5,
                #      act='silu',
                #      )
            ]
        )
        return m

    def load_pretrained_model(self, filename="./weights/darknet53.mix.pth"):
        with open(filename, "rb") as f:
            state_dict = torch.load(f, map_location="cpu")
        print("loading pretrained weights...")
        self.backbone.load_state_dict(state_dict)

    def forward(self, inputs):
        """
        Args:
            inputs (Tensor): input image.

        Returns:
            Tuple[Tensor]: FPN output features..
        """
        #  backbone
        out_features = self.backbone(inputs)
        x3, x2, x1, x0 = [out_features[f] for f in self.in_features]  # 修改：增加了x3
        print("x3的形状：", x3.shape)
        print("x2的形状：", x2.shape)
        print("x1的形状：", x1.shape)
        print("x0的形状：", x0.shape)

        #  yolo branch 1
        x1_in = self.out1_cbl(x0)
        print("x1_in的形状：", x1_in.shape)
        x1_in = self.upsample(x1_in)
        print("x1_in的形状：", x1_in.shape)
        x1_in = torch.cat([x1_in, x1], 1)
        print("x1_in的形状：", x1_in.shape)
        out_dark4 = self.out1(x1_in)
        print("out_dark4的形状：", out_dark4.shape)

        #  yolo branch 2
        x2_in = self.out2_cbl(out_dark4)
        print("x2_in的形状：", x2_in.shape)
        x2_in = self.upsample(x2_in)
        print("x2_in的形状：", x2_in.shape)
        x2_in = torch.cat([x2_in, x2], 1)
        print("x2_in的形状：", x2_in.shape)
        out_dark3 = self.out2(x2_in)
        print("out_dark3的形状：", out_dark3.shape)

        # -------------------------------------#
        #  yolo branch 3  新增加的
        x3_in = self.out3_cbl(out_dark3)
        print("x3_in的形状：", x3_in.shape)
        x3_in = self.upsample(x3_in)
        print("x3_in的形状：", x3_in.shape)
        x3_in = torch.cat([x3_in, x3], 1)
        print("x3_in的形状：", x3_in.shape)
        out_dark2 = self.out3(x3_in)
        print("out_dark2的形状：", out_dark2.shape)
        # -------------------------------------#

        outputs = (out_dark2, out_dark3, out_dark4, x0)  # 修改：增加了out_dark2

        return outputs


if __name__ == "__main__":
    x = torch.rand(1, 3, 640, 640)
    print("x的形状：", x.shape)

    net = YOLOFPN()
    out = net(x)
    # net53 = Darknet(53)
    # out = net53.dark5
    # print(out(x).shape)

    # x = torch.rand(1, 512, 20, 20)
    # print("x的形状：", x.shape)
    #
    # net = C3TR(
    #     512,
    #     512,
    #     n=3,
    #     expansion=0.5,
    #     act='silu',
    # )
    # print(net)
    # out = net(x)
    # print(out.shape)
