import torch
from torch import nn
import torch.nn.functional as F
from transformer import TransformerEncoder,InteractionAttention

import sys

sys.path.append('../')
# from config import *

"""
对不同的模态先进行trans 然后结合 同伴再进行cross attion
之后对不同的模态使用Trans
然后全连接层
"""
modalities_dim = [88, 1024, 512, 714, 139]


class EachFeatureTrans(nn.Module):
    def __init__(self, dim_head, dropout):
        super(EachFeatureTrans, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(88, 512), nn.Dropout(0.2), nn.GELU())
        self.fc2 = nn.Sequential(nn.Linear(1024, 512), nn.Dropout(0.2), nn.GELU())
        self.fc3 = nn.Sequential(nn.Linear(512, 512), nn.Dropout(0.2), nn.GELU())
        self.fc4 = nn.Sequential(nn.Linear(714, 512), nn.Dropout(0.2), nn.GELU())
        self.fc5 = nn.Sequential(nn.Linear(139, 512), nn.Dropout(0.2), nn.GELU())

        self.encode_au1 = TransformerEncoder(num_patches=96, input_dim=512, depth=1, heads=8, mlp_dim=1024, dim_head=dim_head, dropout = dropout)
        self.encode_au2 = TransformerEncoder(num_patches=96, input_dim=512, depth=1, heads=8, mlp_dim=1024, dim_head=dim_head, dropout = dropout)
        self.encode_vi1 = TransformerEncoder(num_patches=96, input_dim=512, depth=1, heads=8, mlp_dim=1024, dim_head=dim_head, dropout = dropout)
        self.encode_vi2 = TransformerEncoder(num_patches=96, input_dim=512, depth=1, heads=8, mlp_dim=1024, dim_head=dim_head, dropout = dropout)
        self.encode_vi3 = TransformerEncoder(num_patches=96, input_dim=512, depth=1, heads=8, mlp_dim=1024, dim_head=dim_head, dropout = dropout)

    def forward(self, inputs):
        au1 = self.fc1(inputs[:, :, :88])
        au1 = self.encode_au1(au1)
        au2 = self.fc2(inputs[:, :, 88:1112])
        au2 = self.encode_au2(au2)
        vi1 = self.fc3(inputs[:, :, 1112:1624])
        vi1 = self.encode_vi1(vi1)
        vi2 = self.fc4(inputs[:, :, 1624:2338])
        vi2 = self.encode_vi2(vi2)
        vi3 = self.fc5(inputs[:, :, 2338:])
        vi3 = self.encode_vi3(vi3)
        au_modality = torch.cat([au1, au2], dim=-1)
        vi_modality = torch.cat([vi1, vi2, vi3], dim=-1)
        return au_modality, vi_modality

class ModalityTrans(nn.Module):
    def __init__(self, dim_head, dropout):
        super(ModalityTrans, self).__init__()
        self.encode_au = TransformerEncoder(num_patches=96, input_dim=1024, depth=1, heads=8, mlp_dim=1024,
                                             dim_head=dim_head, dropout=dropout)
        self.encode_vi = TransformerEncoder(num_patches=96, input_dim=1536, depth=1, heads=8, mlp_dim=1024,
                                             dim_head=dim_head, dropout=dropout)

    def forward(self, au_modality, vi_modality):
        au = self.encode_au(au_modality)
        vi = self.encode_vi(vi_modality)
        return au, vi



class Cross_Block(nn.Module):
    def __init__(self, dim_head, dropout):
        super(Cross_Block, self).__init__()
        self.au_cross = InteractionAttention(num_patches=96, input_dim=1024, depth=1, heads=8, mlp_dim=1024,
                                              dim_head=dim_head, dropout=dropout)
        self.vi_cross = InteractionAttention(num_patches=96, input_dim=1536, depth=1, heads=8, mlp_dim=1024,
                                              dim_head=dim_head, dropout=dropout)

    def forward(self, au1, vi1, au2, vi2):
        au = self.au_cross(au1, au2)
        vi = self.vi_cross(vi1, vi2)
        feature = torch.cat([au, vi], dim=-1)
        return feature


class mymodel(nn.Module):
    def __init__(self):
        super(mymodel, self).__init__()
        self.modalities_dim = [88, 1024, 512, 714, 139]
        self.input_trans = EachFeatureTrans(dim_head=128, dropout=0.3)
        self.partern_trans = EachFeatureTrans(dim_head=128, dropout=0.3)
        self.input_modality_trans = ModalityTrans(dim_head=256, dropout=0.3)
        self.partern_modality_trans = ModalityTrans(dim_head=256, dropout=0.3)
        self.cross_block = Cross_Block(dim_head=256, dropout=0.3)
        self.ff_data = nn.LayerNorm(2477)
        self.ff_partnet_data = nn.LayerNorm(2477)
        self.fc = nn.Sequential(
            nn.LayerNorm(2560),
            nn.Linear(2560, 1),
            nn.SELU()
        )

    def normal_feature(self, inputs):
        normalized_features = []
        start_index = 0
        for dim in self.modalities_dim:
            end_index = start_index + dim
            feature = inputs[:, :, start_index:end_index]
            normalized_feature = F.layer_norm(feature, feature.size()[1:])
            normalized_features.append(normalized_feature)
            start_index = end_index
        x = torch.cat(normalized_features, dim=-1)
        return x

    def forward(self, inputs, partners):
        inputs = self.normal_feature(inputs)
        inputs = self.ff_data(inputs)
        partners = self.normal_feature(partners)
        partners = self.ff_partnet_data(partners)
        au1_modality, vi1_modality = self.input_trans(inputs)
        au2_modality, vi2_modality = self.partern_trans(partners)
        au1_modality, vi1_modality = self.input_modality_trans(au1_modality, vi1_modality)
        au2_modality, vi2_modality = self.partern_modality_trans(au2_modality, vi2_modality)
        feature = self.cross_block(au1_modality, vi1_modality, au2_modality, vi2_modality)
        x = self.fc(feature)
        x = x.squeeze(-1)
        return x


if __name__ == '__main__':
    model = mymodel()
    inputs = torch.randn(4, 96, 2477)
    partners = torch.randn(4, 96, 2477)
    output = model(inputs, partners)
    print(output.shape)
