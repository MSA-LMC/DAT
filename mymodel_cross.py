import torch
from torch import nn
import torch.nn.functional as F
from transformer import TransformerEncoder,InteractionAttention
modalities_dim = [88, 1024, 512, 714, 139]

mlp_dim = 512
dim_head = 128


E1=TransformerEncoder(num_patches=96, input_dim=1024, depth=1, heads=8, mlp_dim=1024, dim_head=dim_head,dropout=0.1)
E2=TransformerEncoder(num_patches=96, input_dim=1536, depth=1, heads=8, mlp_dim=1024, dim_head=dim_head,dropout=0.1)

e1=TransformerEncoder(num_patches=96, input_dim=512, depth=1, heads=4, mlp_dim=mlp_dim, dim_head=dim_head,dropout=0.1)
e2=TransformerEncoder(num_patches=96, input_dim=512, depth=1, heads=4, mlp_dim=mlp_dim, dim_head=dim_head,dropout=0.1)
e3=TransformerEncoder(num_patches=96, input_dim=512, depth=1, heads=4, mlp_dim=mlp_dim, dim_head=dim_head,dropout=0.1)
e4=TransformerEncoder(num_patches=96, input_dim=512, depth=1, heads=4, mlp_dim=mlp_dim, dim_head=dim_head,dropout=0.1)
e5=TransformerEncoder(num_patches=96, input_dim=512, depth=1, heads=4, mlp_dim=mlp_dim, dim_head=dim_head,dropout=0.1)

c=InteractionAttention(num_patches=96, input_dim=768, depth=1, heads=8, mlp_dim=1024, dim_head=dim_head,dropout=0.1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class mymodel(nn.Module):
    def __init__(self, c=c,E1=E1,E2=E2,e1=e1,e2=e2,e3=e3,e4=e4,e5=e5):
        super(mymodel, self).__init__()
        self.cross_att = c.to(device)#交叉注意力特征融合编码器
        self.E1 = E1.to(device)#音频特征编码器
        self.E2 = E2.to(device)#视觉特征编码器

        #各个特征的编码器
        self.encoder1 = e1.to(device)
        self.encoder2 = e2.to(device)
        self.encoder3 = e3.to(device)
        self.encoder4 = e4.to(device)
        self.encoder5 = e5.to(device)

        self.fc1 = nn.Linear(88, 512)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(714, 512)
        self.fc5 = nn.Linear(139, 512)

        self.fx1=nn.Linear(1024,768)
        self.fx2=nn.Linear(1536,768)



        self.fc = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, 1),
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
        #将所有归一化后的特征沿着特定维度连接起来
        x=torch.cat(normalized_features, dim=-1)

        #整体归一化 (不知道有没有用，可以消融一下试试）
        x=self.ff(x)

        #音频特征
        vectors = []
        split_inputs = split_features(x, modalities_dim)
        for i in [0, 1]:
            fc = getattr(self, f'fc{i + 1}')            # 动态获取线性映射层
            encoder = getattr(self, f'encoder{i + 1}')  # 动态获取编码器
            feature=fc(split_inputs[i])
            feature = encoder(feature)  # 使用编码器处理输入
            vectors.append(feature)
        x1 = torch.cat(vectors, dim=-1)     #(batchsizes,96,1024（512*2）)
        x1=self.E1(x1)                      #(batchsizes,96,1024)
        x1=self.fx1(x1) #统一映射到768维度     #(batchsizes,96,768)

        #视觉特征
        vectors = []
        split_inputs = split_features(x, modalities_dim)
        for i in [2, 3, 4]:
            fc = getattr(self, f'fc{i + 1}')            # 动态获取线性映射层
            encoder = getattr(self, f'encoder{i + 1}')  # 动态获取编码器
            feature = fc(split_inputs[i])
            feature = encoder(feature)  # 使用编码器处理输入
            vectors.append(feature)

        x2 = torch.cat(vectors, dim=-1)     #(batchsizes,96,1536(512*3))
        x2=self.E2(x2)                      #(batchsizes,96,1536)
        x2=self.fx2(x2) #统一映射到768维度     #(batchsizes,96,768)

        #交叉注意力进行特征融合
        y=self.cross_att(x1,x2)             #(batchsizes,96,768)
        y = self.fc(y)                      #(batchsizes,96,1)
        y = y.squeeze(-1)                   #(batchsizes,96)
        return y

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
#
model=model.to(device)
inputs,par_inputs1,par_inputs2,par_inputs3=inputs.to(device),par_inputs1.to(device),par_inputs2.to(device),par_inputs3.to(device)
#
out=model(inputs)
print(out.size())'''