# Ross3D: Reconstructive Visual Instruction Tuning with 3D-Awareness

[[Project Page](https://haochen-wang409.github.io/ross3d/)] [[Model Zoo (Coming Soon)](https://huggingface.co/HaochenWang/)]

[**Ross3D: Reconstructive Visual Instruction Tuning with 3D-Awareness**](https://arxiv.org/pdf/2504.01901) by
[Haochen Wang](https://haochen-wang409.github.io), 
[Yucheng Zhao](https://scholar.google.com/citations?user=QWemjjQAAAAJ&hl=en),
[Tiancai Wang](https://scholar.google.com/citations?user=YI0sRroAAAAJ&hl=en),
[Haoqiang Fan](https://scholar.google.com/citations?hl=en),
[Xiangyu Zhang](https://scholar.google.com/citations?user=yuB-cfoAAAAJ&hl=en), and
[Zhaoxiang Zhang](https://scholar.google.com/citations?user=qxWfV6cAAAAJ).

> **Abstract.** 
> The rapid development of Large Multimodal Models (LMMs) for 2D images and videos has
> spurred efforts to adapt these models for interpreting 3D scenes. 
> However, the absence of large-scale 3D vision-language datasets 
> has posed a significant obstacle. To address this issue, 
> typical approaches focus on injecting 3D awareness into 
> 2D LMMs by designing 3D input-level scene representations. 
> This work provides a new perspective. We introduce 
> reconstructive visual instruction tuning with 3D-awareness (Ross3D), 
> which integrates 3D-aware visual supervision 
> into the training procedure. Specifically, it incorporates 
> cross-view and global-view reconstruction. The former requires 
> reconstructing masked views by aggregating overlapping information
> from other views. The latter aims to aggregate information from all 
> available views to recover Bird’s-Eye-View images, contributing to a comprehensive overview
> of the entire scene. Empirically, ROSS3D achieves state-of-the-art 
> performance across various 3D scene understanding benchmarks. 
> More importantly, our semi-supervised experiments demonstrate significant potential in leveraging
> large amounts of unlabeled 3D vision-only data.

![](./img/method.png)

The code will be released soon!
