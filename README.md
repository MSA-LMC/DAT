# DAT: Dialogue-Aware Transformer with Modality-Group Fusion for Human Engagement Estimation，MM 2024

## Task Description

This project aims to estimate human engagement in conversations through audio and visual input. Engagement estimation is a key indicator for understanding human social behavior, widely applied in emotion computing and human-computer interaction fields.
![Human Engagement Estimation](https://github.com/MSA-LMC/DAT/blob/main/data_vis.jpg)
This photo shows the NOXI dataset (upper part) and the MPIIGroupInteraction dataset (lower part)[1].


## 方法

我们提出了一种Dialogue-aware Transformer (DAT) 框架，结合了模态组融合 (Modality-Group Fusion, MGF) 策略，以提高对话中参与者的参与度估计准确性。我们的方法主要包括以下几个关键组件：

1. **Modality-Group Fusion (MGF)**：
   - 该模块独立融合每个参与者的音频和视觉特征，生成深层、语言独立的特征表示。
   - 通过减少冗余信息，优化模态间的特征整合，提升模型性能和鲁棒性。

2. **Dialogue-Aware Transformer Encoder (DAE)**：
   - 利用交叉注意力机制，结合来自目标参与者和对话伙伴的音视频特征，提高参与度估计的准确性。
   - DAE 模块通过整合额外的上下文信息，有效提升了模型对目标参与者行为的捕获能力。

3. **框架架构**：
   - DAT 框架由多个 Transformer 层组成，包括 MG 和 DAE 模块，通过对参与者的音频-视觉内容进行建模，进行行为分析。

## 实验效果

我们在 MultiMediate 2024 组织的多域参与度估计挑战赛中严格测试了该方法，结果显示：

- 在 NoXi Base 测试集上，模型达到了 0.76 的 CCC（Concordance Correlation Coefficient）得分。
- 在 NoXi-Add 和 MPIIGI 测试集上的平均 CCC 分别为 0.67 和 0.49。
- 综合各测试集的平均CCC得分提升了 0.23，相较于基线模型显著提高了模型的表现。

### 结果总结

本研究显著相较于基线模型提升了参与度估计的精度和鲁棒性，证明了对话伙伴信息和模态组融合策略在模型中的重要性。

## 参考文献

如需了解更多细节，可以参考以下文献：
- Li, J., Yu, Y., Chen, Y., Zhang, Y., Jia, P., Xu, Y., Li, Z., Wang, M., Hong, R. (2024). DAT: Dialogue-Aware Transformer with Modality-Group Fusion for Human Engagement Estimation. [arXiv preprint](https://arxiv.org/abs/xxxx.xxxxx).



