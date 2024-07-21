import torch
from torch import nn
import torch.nn.functional as F
from transformer import TransformerEncoder,InteractionAttention
modalities_dim = [88, 1024, 512, 714, 139]


e=TransformerEncoder(num_patches=96, input_dim=2477, depth=1, heads=8, mlp_dim=1024, dim_head=64)
e1=TransformerEncoder(num_patches=96, input_dim=88, depth=1, heads=4, mlp_dim=256, dim_head=64)
e2=TransformerEncoder(num_patches=96, input_dim=1024, depth=1, heads=4, mlp_dim=256, dim_head=64)
e3=TransformerEncoder(num_patches=96, input_dim=512, depth=1, heads=4, mlp_dim=256, dim_head=64)
e4=TransformerEncoder(num_patches=96, input_dim=714, depth=1, heads=4, mlp_dim=256, dim_head=64)
e5=TransformerEncoder(num_patches=96, input_dim=139, depth=1, heads=4, mlp_dim=256, dim_head=64)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class mymodel(nn.Module):
    def __init__(self, e=e, e1=e1,e2=e2, e3=e3, e4=e4, e5=e5):
        super(mymodel, self).__init__()
        self.encoder = e.to(device)
        self.encoder1 = e1.to(device)
        self.encoder2 = e2.to(device)
        self.encoder3 = e3.to(device)
        self.encoder4 = e4.to(device)
        self.encoder5 = e5.to(device)

        self.fc = nn.Sequential(
            nn.LayerNorm(2477),
            nn.Linear(2477, 1),
            nn.SELU(),
        )
        self.ff=nn.LayerNorm(2477)

    def forward(self, inputs):
        normalized_features = []
        start_index = 0
        for dim in modalities_dim:
            end_index = start_index + dim
            # 提取当前特征
            feature = inputs[:, :, start_index:end_index]
            # 对当前特征进行 Layer Normalization
            normalized_feature = F.layer_norm(feature, feature.size()[1:])
            # 将归一化后的特征添加到列表中
            normalized_features.append(normalized_feature)
            start_index = end_index

        # 将所有归一化后的特征沿着特定维度连接起来
        x=torch.cat(normalized_features, dim=-1)
        x=self.ff(x)
        vectors = []
        split_inputs = split_features(x, modalities_dim)  # 分成(batchsize,)
        for i in [0,1,2,3,4]:
            encoder = getattr(self, f'encoder{i + 1}')  # 动态获取编码器
            feature = encoder(split_inputs[i])  # 使用编码器处理输入
            vectors.append(feature)
        x = torch.cat(vectors, dim=-1)
        x = self.encoder(x)
        x = self.fc(x)
        x = x.squeeze(-1)
        return x

def split_features(inputs, modalities_dim):
    batchsize, timesteps, total_dim = inputs.shape
    assert total_dim == sum(modalities_dim), "总维度与各个特征维度之和不匹配"

    split_features = []
    start_idx = 0
    for dim in modalities_dim:
        split_features.append(inputs[:, :, start_idx:start_idx + dim])
        start_idx += dim

    return split_features

'''model=mymodel()
inputs=torch.rand(32,96,2477)
par_inputs1=torch.rand(32,96,2477)
par_inputs2=torch.rand(32,96,2477)
par_inputs3=torch.rand(32,96,2477)

model=model.to(device)
inputs,par_inputs1,par_inputs2,par_inputs3=inputs.to(device),par_inputs1.to(device),par_inputs2.to(device),par_inputs3.to(device)

out=model(inputs)
print(out.size())'''