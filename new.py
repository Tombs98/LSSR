import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.arch_utils import LayerNorm2d, MySequential





class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

## Combination Coefficient
class CC(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CC, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_mean = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )
        self.conv_std = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):

        # mean
        ca_mean = self.avg_pool(x)
        ca_mean = self.conv_mean(ca_mean)

        # std
        m_batchsize, C, height, width = x.size()
        x_dense = x.view(m_batchsize, C, -1)
        ca_std = torch.std(x_dense, dim=2, keepdim=True)
        ca_std = ca_std.view(m_batchsize, C, 1, 1)
        ca_var = self.conv_std(ca_std)


        cc = (ca_mean + ca_var)/2.0
        return cc
    

class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma

class LatticeBlock(nn.Module):   
    def __init__(self, c, DW_Expand=2, drop_out_rate=0.):
        super().__init__()

        dw_channel = c * DW_Expand

        # block_0 = []
        # block_0.append(LayerNorm2d(c))
        # block_0.append(nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True))
        # block_0.append(nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
        #                        bias=True))
        # block_0.append(SimpleGate())
        # block_0.append(nn.AdaptiveAvgPool2d(1))
        # block_0.append(nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
        #               groups=1, bias=True))
        # block_0.append(nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True))

        self.conv_block0 = NAFBlock(c)

        self.fea_ca1 = CC(c)
        self.x_ca1 = CC(c)
        
        # block_1 = []
        # block_1.append(LayerNorm2d(c))
        # block_1.append(nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True))
        # block_1.append(SimpleGate())
        # block_1.append(nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True))

        self.conv_block1 = NAFBlock(c)
        

        self.fea_ca2 = CC(c)
        self.x_ca2 = CC(c)

        self.compress = nn.Conv2d(2 * c, c, kernel_size=1, padding=0, bias=True)


    def forward(self, x):

        x_feature_shot = self.conv_block0(x)
        fea_ca1 = self.fea_ca1(x_feature_shot)
        x_ca1 = self.x_ca1(x)

        p1z = x + fea_ca1 * x_feature_shot
        q1z = x_feature_shot + x_ca1 * x

        # synthes_unit
        x_feat_long = self.conv_block1(p1z)
        fea_ca2 = self.fea_ca2(q1z)
        p3z = x_feat_long + fea_ca2 * q1z
        x_ca2 = self.x_ca2(x_feat_long)
        q3z = q1z + x_ca2 * x_feat_long

        out = torch.cat((p3z, q3z), 1)
        out = self.compress(out)

        return out
    
class LatticeBlock2(nn.Module):
    def __init__(self, nFeat):
        super(LatticeBlock2, self).__init__()

    
        block_0 = []
        block_0.append(nn.Conv2d(nFeat, nFeat, 3, 1, 1, groups=4, bias=True))
        block_0.append(nn.LeakyReLU(0.1, inplace=True))
        block_0.append(nn.Conv2d(nFeat, nFeat, 3, 1, 1, groups=4, bias=True))
        self.conv_block0 = nn.Sequential(*block_0)

        self.fea_ca1 = CC(nFeat)
        self.x_ca1 = CC(nFeat)

        block_1 = []
        block_1.append(nn.Conv2d(nFeat, nFeat, 3, 1, 1, groups=4, bias=True))
        block_1.append(nn.LeakyReLU(0.1, inplace=True))
        block_1.append(nn.Conv2d(nFeat, nFeat, 3, 1, 1, groups=4, bias=True))
        self.conv_block1 = nn.Sequential(*block_1)

        self.fea_ca2 = CC(nFeat)
        self.x_ca2 = CC(nFeat)

        self.compress = nn.Conv2d(2 * nFeat, nFeat, kernel_size=1, padding=0, bias=True)

    def forward(self, x):
        # analyse unit
        x_feature_shot = self.conv_block0(x)
        fea_ca1 = self.fea_ca1(x_feature_shot)
        x_ca1 = self.x_ca1(x)

        p1z = x + fea_ca1 * x_feature_shot
        q1z = x_feature_shot + x_ca1 * x

        # synthes_unit
        x_feat_long = self.conv_block1(p1z)
        fea_ca2 = self.fea_ca2(q1z)
        p3z = x_feat_long + fea_ca2 * q1z
        x_ca2 = self.x_ca2(x_feat_long)
        q3z = q1z + x_ca2 * x_feat_long

        out = torch.cat((p3z, q3z), 1)
        out = self.compress(out)

        return out
 

class SCAM(nn.Module):
    '''
    Stereo Cross Attention Module (SCAM)
    '''
    def __init__(self, c):
        super().__init__()
        self.scale = c ** -0.5

        self.norm_l = LayerNorm2d(c)
        self.norm_r = LayerNorm2d(c)

        self.lb1 = LatticeBlock(c)
        self.l_proj1 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)
        self.r_proj1 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)
        
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.lb2 = LatticeBlock(c)
        self.l_proj2 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)
        self.r_proj2 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)

    def forward(self, x_l, x_r):
        
        Q_l = self.l_proj1(self.lb1(self.norm_l(x_l)))
        Q_r = self.r_proj1(self.lb2(self.norm_r(x_r)))
        b, c, h, w = Q_r.shape
        Q_l = Q_l - torch.mean(Q_l, 3).unsqueeze(3).repeat(1, 1, 1, w)
        Q_r = Q_r - torch.mean(Q_r, 3).unsqueeze(3).repeat(1, 1, 1, w)
        Q_l = Q_l.permute(0, 2, 3, 1)
        Q_r_T = Q_r.permute(0, 2, 1, 3)

        V_l = self.l_proj2(x_l).permute(0, 2, 3, 1)  # B, H, W, c
        V_r = self.r_proj2(x_r).permute(0, 2, 3, 1)  # B, H, W, c


        # (B, H, W, c) x (B, H, c, W) -> (B, H, W, W)
        attention = torch.matmul(Q_l, Q_r_T) * self.scale

        F_r2l = torch.matmul(torch.softmax(attention, dim=-1), V_r)  #B, H, W, c
        F_l2r = torch.matmul(torch.softmax(attention.permute(0, 1, 3, 2), dim=-1), V_l) #B, H, W, c

        # scale
        F_r2l = F_r2l.permute(0, 3, 1, 2) * self.beta
        F_l2r = F_l2r.permute(0, 3, 1, 2) * self.gamma
        return x_l + F_r2l, x_r + F_l2r


class St(nn.Module):
    def __init__(self, c, drop_out_rate=0., fusion=False):
        super().__init__()
        self.blk = LatticeBlock(c, drop_out_rate=drop_out_rate)
        self.fusion = SCAM(c) if fusion else None

    def forward(self, *feats):
        feats = tuple([self.blk(x) for x in feats])
        if self.fusion:
            feats = self.fusion(*feats)
        return feats


class DropPath(nn.Module):
    def __init__(self, drop_rate, module):
        super().__init__()
        self.drop_rate = drop_rate
        self.module = module

    def forward(self, *feats):
        if self.training and np.random.rand() < self.drop_rate:
            return feats

        new_feats = self.module(*feats)
        factor = 1. / (1 - self.drop_rate) if self.training else 1.

        if self.training and factor != 1.:
            new_feats = tuple([x+factor*(new_x-x) for x, new_x in zip(feats, new_feats)])
        return new_feats



class NAFNetSR(nn.Module):
    '''
    NAFNet for Super-Resolution
    '''
    def __init__(self, up_scale=4, width=48, num_blks=8, img_channel=3, drop_path_rate=0., drop_out_rate=0.,fusion_from=-1, fusion_to=1000, dual=True):
        super().__init__()
        self.dual = dual    # dual input for stereo SR (left view, right view)
        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        

        self.body = MySequential(
            *[DropPath(
                drop_path_rate, 
                St(
                    width, 
                    fusion=(fusion_from <= i and i <= fusion_to), 
                    drop_out_rate=drop_out_rate
                )) for i in range(num_blks)]
        )
        self.up = nn.Sequential(
            nn.Conv2d(in_channels=width, out_channels=img_channel * up_scale**2, kernel_size=3, padding=1, stride=1, groups=1, bias=True),
            nn.PixelShuffle(up_scale)
        )
        self.upscale_factor = up_scale
        # self.dr = drop_path_rate

    def forward(self, inp):
        # print(self.dr)
        inp_hr = F.interpolate(inp, scale_factor=self.upscale_factor, mode='bilinear')
        if self.dual:
            inp = inp.chunk(2, dim=1)
        else:
            inp = (inp, )
        
        feats = [self.intro(x) for x in inp]
        feats = self.body(*feats)
        out = torch.cat([self.up(x) for x in feats], dim=1)
        out = out + inp_hr
        return out
        


class AvgPool2d(nn.Module):
    def __init__(self, kernel_size=None, base_size=None, auto_pad=True, fast_imp=False, train_size=None):
        super().__init__()
        self.kernel_size = kernel_size
        self.base_size = base_size
        self.auto_pad = auto_pad

        # only used for fast implementation
        self.fast_imp = fast_imp
        self.rs = [5, 4, 3, 2, 1]
        self.max_r1 = self.rs[0]
        self.max_r2 = self.rs[0]
        self.train_size = train_size

    def extra_repr(self) -> str:
        return 'kernel_size={}, base_size={}, stride={}, fast_imp={}'.format(
            self.kernel_size, self.base_size, self.kernel_size, self.fast_imp
        )

    def forward(self, x):
        if self.kernel_size is None and self.base_size:
            train_size = self.train_size
            if isinstance(self.base_size, int):
                self.base_size = (self.base_size, self.base_size)
            self.kernel_size = list(self.base_size)
            self.kernel_size[0] = x.shape[2] * self.base_size[0] // train_size[-2]
            self.kernel_size[1] = x.shape[3] * self.base_size[1] // train_size[-1]

            # only used for fast implementation
            self.max_r1 = max(1, self.rs[0] * x.shape[2] // train_size[-2])
            self.max_r2 = max(1, self.rs[0] * x.shape[3] // train_size[-1])

        if self.kernel_size[0] >= x.size(-2) and self.kernel_size[1] >= x.size(-1):
            return F.adaptive_avg_pool2d(x, 1)

        if self.fast_imp:  # Non-equivalent implementation but faster
            h, w = x.shape[2:]
            if self.kernel_size[0] >= h and self.kernel_size[1] >= w:
                out = F.adaptive_avg_pool2d(x, 1)
            else:
                r1 = [r for r in self.rs if h % r == 0][0]
                r2 = [r for r in self.rs if w % r == 0][0]
                # reduction_constraint
                r1 = min(self.max_r1, r1)
                r2 = min(self.max_r2, r2)
                s = x[:, :, ::r1, ::r2].cumsum(dim=-1).cumsum(dim=-2)
                n, c, h, w = s.shape
                k1, k2 = min(h - 1, self.kernel_size[0] // r1), min(w - 1, self.kernel_size[1] // r2)
                out = (s[:, :, :-k1, :-k2] - s[:, :, :-k1, k2:] - s[:, :, k1:, :-k2] + s[:, :, k1:, k2:]) / (k1 * k2)
                out = torch.nn.functional.interpolate(out, scale_factor=(r1, r2))
        else:
            n, c, h, w = x.shape
            s = x.cumsum(dim=-1).cumsum_(dim=-2)
            s = torch.nn.functional.pad(s, (1, 0, 1, 0))  # pad 0 for convenience
            k1, k2 = min(h, self.kernel_size[0]), min(w, self.kernel_size[1])
            s1, s2, s3, s4 = s[:, :, :-k1, :-k2], s[:, :, :-k1, k2:], s[:, :, k1:, :-k2], s[:, :, k1:, k2:]
            out = s4 + s1 - s2 - s3
            out = out / (k1 * k2)

        if self.auto_pad:
            n, c, h, w = x.shape
            _h, _w = out.shape[2:]
            # print(x.shape, self.kernel_size)
            pad2d = ((w - _w) // 2, (w - _w + 1) // 2, (h - _h) // 2, (h - _h + 1) // 2)
            out = torch.nn.functional.pad(out, pad2d, mode='replicate')

        return out


def replace_layers(model, base_size, train_size, fast_imp, **kwargs):
    for n, m in model.named_children():
        if len(list(m.children())) > 0:
            ## compound module, go inside it
            replace_layers(m, base_size, train_size, fast_imp, **kwargs)

        if isinstance(m, nn.AdaptiveAvgPool2d):
            pool = AvgPool2d(base_size=base_size, fast_imp=fast_imp, train_size=train_size)
            assert m.output_size == 1
            setattr(model, n, pool)

        # if isinstance(m, Attention):
        #     attn = LocalAttention(dim=m.dim, num_heads=m.num_heads, is_prompt=m.is_prompt, bias=True, base_size=base_size, fast_imp=False,
        #                           train_size=train_size)
        #     setattr(model, n, attn)


class Local_Base():
    def convert(self, *args, train_size, **kwargs):
        replace_layers(self, *args, train_size=train_size, **kwargs)
        imgs = torch.rand(train_size)
        with torch.no_grad():
            self.forward(imgs)


class NAFSSR(Local_Base, NAFNetSR):
    def __init__(self, *args, train_size=(1, 6, 30, 90), fast_imp=False, fusion_from=-1, fusion_to=1000, **kwargs):
        Local_Base.__init__(self)
        NAFNetSR.__init__(self, *args, img_channel=3, fusion_from=fusion_from, fusion_to=fusion_to, dual=True, **kwargs)

        N, C, H, W = train_size
        base_size = (int(H * 1.5), int(W * 1.5))

        self.eval()
        with torch.no_grad():
            self.convert(base_size=base_size, train_size=train_size, fast_imp=fast_imp)


if __name__ == "__main__":
    net = NAFNetSR()
    x1 =  torch.randn((1, 3, 30, 90))
    x2 =  torch.randn((1, 3, 30, 90))
    x = torch.cat((x1,x2),dim=1)
    print("Total number of param  is ", sum(i.numel() for i in net.parameters()))
    # t=net(x)
    # print(len(t))
    inp_shape = (6, 64, 64)
    from ptflops import get_model_complexity_info
    FLOPS = 0

    macs, params = get_model_complexity_info(net, inp_shape, verbose=False, print_per_layer_stat=True)
    # print(params)
    macs = float(macs[:-4]) + FLOPS / 10 ** 9



    print('mac', macs, params)

    from thop import profile
    x3 =  torch.randn((1, 6, 128, 128))
    flops, params = profile(net, inputs=(x3, ))
    print('FLOPs = ' + str(flops/1000**3) + 'G')
    print('Params = ' + str(params/1000**2) + 'M')

   