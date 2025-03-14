# VisualWebInstruct
The official repo for [VisualWebInstruct: Scaling up Multimodal Instruction Data through Web Search](https://arxiv.org/abs/2503.10582).

## Introduction
Vision-Language Models have made significant progress on many perception-focused tasks, however, their progress on reasoning-focused tasks seem to be limited due to the lack of high-quality and diverse training data. In this work, we aim to address the scarcity issue of reasoning-focused multimodal datasets. We propose VisualWebInstruct - a novel approach that leverages search engine to create a diverse, and high-quality dataset spanning multiple disciplines like math, physics, finance, chemistry, etc. Starting with meticulously selected 30,000 seed images, we employ Google Image search to identify websites containing similar images. We collect and process the HTMLs from over 700K unique URL sources. Through a pipeline of content extraction, filtering and synthesis, we build a dataset of approximately 900K question-answer pairs, with 40% being visual QA pairs and the rest as text QA pairs. Models fine-tuned on VisualWebInstruct demonstrate significant performance gains: (1) training from Llava-OV-mid shows 10-20% absolute point gains across benchmarks, (2) training from MAmmoTH-VL shows 5% absoluate gain. Our best model MAmmoTH-VL2 shows state-of-the-art performance within the 10B parameter class on MMMU-Pro-std (40.7%), MathVerse (42.6%), and DynaMath (55.7%). These remarkable results highlight the effectiveness of our dataset in enhancing VLMs' reasoning capabilities for complex multimodal tasks.


## Citation
```
@article{visualwebinstruct,
    title={VisualWebInstruct: Scaling up Multimodal Instruction Data through Web Search},
    author = {Jia, Yiming and Li, Jiachen and Yue, Xiang and Li, Bo and Nie, Ping and Zou, Kai and Chen, Wenhu},
    journal={arXiv preprint arXiv:2503.10582},
    year={2025}
}
```
