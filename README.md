# RHMNet-for-SSVOS

This paper aims to solve the video object segmentation (VOS) task in a scribble-supervised manner, in which VOS models are not only trained by the sparse scribble annotations but also initialized with the sparse target scribbles for inference. Thus, the annotation burdens for both training and initialization can be substantially lightened. The difficulties of scribble-supervised VOS lie in two aspects. On the one hand, it requires the powerful ability to learn from the sparse scribble annotations during training. On the other hand, it demands strong reasoning capability during inference given only a sparse initial target scribble. In this work, we propose a Reliability-Hierarchical Memory Network (RHMNet) to predict the target mask in a step-wise expanding strategy w.r.t. the memory reliability level. To be specific, RHMNet first only uses the memory in the high reliability level to locate the region with high reliability belonging to the target, which is highly similar to the initial target scribble. Then it expands the located high-reliability region to the entire target conditioned on the region itself and the memories in all reliability levels. Besides, we propose a scribble-supervised learning mechanism to facilitate the learning of our model to predict dense results. It mines the pixel-level relation within the single frame and the frame-level relation within the sequence to take full advantage of the scribble annotations in sequence training samples. The favorable performance on two popular benchmarks demonstrates that our method is promising.
> **Reliablity-Hierarchical Memory Network for Scribble-Supervised Video Object Segmentation**, Zikun Zhou, Kaige Mao, Wenjie Pei, Hongpeng Wang, Yaowei Wang, Zhenyu He. [[Paper]()]

## Framework of our RHMNet

![Overall framework of our Reliability-Hierarchical Memory Network](figs/framework.jpg)
Our Reliability-Hierarchical Memory Network consists of the reliability-hierarchical memory bank, the feature extractor, the memory encoding module, the matching module, and the segmentation head. For processing a new frame, RHMNet first captures the reliable region, which is the region highly similar to the initial target scribble region, and then accordingly segments the entire target. In each expanding step, only the historical information in the corresponding or higher reliability level is used as the reference for memory matching.

## Scribble Samples

Synthesized training scribbles

Manully drawn evaluation scribbles

- DAVIS
![Evaluation scribbles for DAVIS](figs/evaluation_scribbles_for_davis.png)
- Youtube-VOS
![Evaluation scribbles for Youtube-VOS](figs/evaluation_scribbles_for_ytb.png)

## Download the Scribble Annotations

The synthesized scribble annotations used for training could be download from here: [[GoogleDriver](https://drive.google.com/drive/folders/1VbRXU9PZjSJ4qQDGRapGXGAsB6XhpLru?usp=sharing)] [[BaiduYun(code:o2ey)](https://pan.baidu.com/s/1_8fFs1QP2Pta-DXrqzmWrA)]

The manully drawn initial scribbles for the validation set of DAVIS and Youtube-VOS could be download from here:  [[GoogleDriver](https://drive.google.com/file/d/17vQd9jmhprWflIsy8kntSOf-d51RPZWL/view?usp=sharing)] [[BaiduYun(code:yhdj)](https://pan.baidu.com/s/1ir6h3RGa1uRiI6XTIdMREw)]

In the annotation file of a frame with $N$ objects, the region with a value of $n$ ($n\in\{1, 2, ..., N\}$) means the scribble of the $n$-th object, the region with a value of $N+1$ means the scribble of the background, and the region with a value of $0$ means the unlabeled region.

## Training and Evaluation

The code and pre-trained model is coming soon.
