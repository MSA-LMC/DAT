      # DAT: Dialogue-Aware Transformer with Modality-Group Fusion for Human Engagement Estimation，MM 2024

## Task Description

This project aims to estimate human engagement in conversations through audio and visual input. Engagement estimation is a key indicator for understanding human social behavior, widely applied in emotion computing and human-computer interaction fields.

<img src="https://github.com/your_username/your_repo_name/images/data_vis.jpg" alt="Human Engagement Estimation" width="600"/>

This photo displays an overview of the NOXI dataset (upper part) and the MPIIGroupInteraction dataset (lower part). For specifics, please refer to [1].


## Method       

We propose a Dialogue-Aware Transformer (DAT) framework that incorporates a Modality-Group Fusion (MGF) strategy to enhance the accuracy of engagement estimation in conversations.

<img src="https://github.com/your_username/your_repo_name/images/model_structure.png" alt="Model Structure" width="600"/>

## 实验效果

我们在 MultiMediate 2024 组织的多域参与度估计挑战赛中严格测试了该方法，结果显示：

- 在 NoXi Base 测试集上，模型达到了 0.76 的 CCC（Concordance Correlation Coefficient）得分。
- 在 NoXi-Add 和 MPIIGI 测试集上的平均 CCC 分别为 0.67 和 0.49。
- 综合各测试集的平均CCC得分提升了 0.23，相较于基线模型显著提高了模型的表现。

### 结果总结

本研究显著相较于基线模型提升了参与度估计的精度和鲁棒性，证明了对话伙伴信息和模态组融合策略在模型中的重要性。

## Reference

[1] Philipp Müller, Michal Balazia, Tobias Baur, Michael Dietz, Alexander Heimerl, Dominik Schiller, Mohammed Guermal, Dominike Thomas, François Brémond, Jan Alexandersson, Elisabeth André, and Andreas Bulling. 2023. MultiMediate '23: Engagement Estimation and Bodily Behaviour Recognition in Social Interactions. In Proceedings of the 31st ACM International Conference on Multimedia (MM '23). Association for Computing Machinery, New York, NY, USA, 9640–9645. https://doi.org/10.1145/3581783.3613851




