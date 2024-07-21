import torch
import torch.nn as nn

class CenterMSELoss(nn.Module):
    def __init__(self, reduction='none', beta=0.5):
        super(CenterMSELoss, self).__init__()
        self.beta = beta
        self.MSE_loss = nn.MSELoss(reduction=reduction)

    def forward(self, outputs, labels):
        mse_loss = self.MSE_loss(outputs, labels)

        #这是一个seq2seq的模型，所以最终的输出为（batch_size, seq_num),这里的length也就是seq_num
        length = outputs.size(1)

        weights = torch.cat([
            torch.ones(length // 3) * self.beta,
            torch.ones(length // 3),
            torch.ones(length - 2 * (length // 3)) * self.beta
        ]).to(outputs.device)

        #首先unsqueeze(0):0表示再最前面，加一个维度，也就是再最前面升一维
        #expand(size,-1): 表示第一个维度上扩展为size,第二个维度为-1，表示不变
        weights = weights.unsqueeze(0).expand(outputs.shape[0], -1)


        #可以试试这两个损失函数有什么区别（感觉显性的损失更加有效）
        weighted_loss = torch.mean(mse_loss * weights) # explicitly

        # weighted_loss = （self.beta+1+self.beta) / 3 * weighted_loss  # implicitly

        return weighted_loss

