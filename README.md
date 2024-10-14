# [MM 2024] DAT: Dialogue-Aware Transformer with Modality-Group Fusion for Human Engagement Estimation

## Task Description

This project aims to estimate human engagement in conversations through audio and visual input. Engagement estimation is a key indicator for understanding human social behavior, widely applied in emotion computing and human-computer interaction fields.

<img src="https://github.com/MSA-LMC/DAT/blob/main/data_vis.jpg" alt="Human Engagement Estimation" width="400"/>


This photo displays an overview of the NOXI dataset (upper part) and the MPIIGroupInteraction dataset (lower part). For specifics, please refer to [1].


## Method       

We propose a Dialogue-Aware Transformer (DAT) framework to enhance the accuracy of engagement estimation in conversations.

<img src="https://github.com/MSA-LMC/DAT/blob/main/model_structure.jpg" alt="Model Structure" width="1500"/>

**Overall architecture of the proposed method**.  Our DAT consists of two main modules: _Modality-Group Fusion_ and _Dialogue-Aware Encoder_. Firstly, the Modality-Group Fusion module processes audio and visual features for both the participant and partner. Each feature is processed through a Transformer before being fused together. Subsequently, the Dialogue-Aware Encoder utilizes cross-attention to combine and encode information from both participants, focusing on contextual interactions to enhance engagement prediction. Finally, an MLP predicts continuous engagement levels frame-by-frame by utilizing the encoded features.

## Result

We rigorously tested this method at the MultiMediate 2024 multi-domain engagement estimation challenge, achieving the best performance on the NoXi Base dataset and the 3rd place across three different datasets (multi-domain):
| Team              | NoXi Base | NoXi Add | MPIIGI | Global |
|-------------------|-----------|----------|--------|--------|
| USTC-IAT-Unite    | 0.72      | 0.73     | 0.59   | 0.68   |
| AI-lab            | 0.69      | 0.72     | 0.54   | 0.65   |
| **HFUT-LMC (Ours)**| **0.76** | **0.67** | **0.49** | **0.64** |
| Syntax            | 0.72      | 0.69     | 0.5    | 0.64   |
| ashk              | 0.72      | 0.69     | 0.42   | 0.61   |
| YKK               | 0.68      | 0.66     | 0.36   | 0.54   |
| Xpace             | 0.7       | 0.7      | 0.34   | 0.58   |
| nox               | 0.68      | 0.66     | 0.35   | 0.57   |
| SP-team           | 0.68      | 0.65     | 0.34   | 0.56   |
| YI.YJ             | 0.6       | 0.52     | 0.3    | 0.47   |
| MM24 Baseline     | 0.64      | 0.51     | 0.09   | 0.41   |

## Dataset and Reference

[1] Philipp Müller, Michal Balazia, Tobias Baur, Michael Dietz, Alexander Heimerl, Dominik Schiller, Mohammed Guermal, Dominike Thomas, François Brémond, Jan Alexandersson, Elisabeth André, and Andreas Bulling. 2023. MultiMediate '23: Engagement Estimation and Bodily Behaviour Recognition in Social Interactions. In Proceedings of the 31st ACM International Conference on Multimedia (MM '23).


## Citation
If you find this repo helpful, please consider citing:

```
@misc{li2024datdialogue,
      title={DAT: Dialogue-Aware Transformer with Modality-Group Fusion for Human Engagement Estimation}, 
      author={Jia Li and Yangchen Yu and Yin Chen and Yu Zhang and Peng Jia and Yunbo Xu and Ziqiang Li and Meng Wang and Richang Hong},
      year={2024},
      eprint={2410.08470},
      archivePrefix={arXiv},
      primaryClass={cs.HC},
      url={https://arxiv.org/abs/2410.08470}, 
}
```

