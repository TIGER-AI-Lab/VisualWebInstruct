# VisualWebInstruct
The official repo for [VisualWebInstruct: Scaling up Multimodal Instruction Data through Web Search](https://arxiv.org/abs/2503.10582).

<a target="_blank" href="https://arxiv.org/abs/2503.10582">
<img style="height:22pt" src="https://img.shields.io/badge/-Paper-black?style=flat&logo=arxiv">
</a>

<a target="_blank" href="https://huggingface.co/datasets/TIGER-Lab/VisualWebInstruct">
<img style="height:22pt" src="https://img.shields.io/badge/-🤗%20Dataset-red?style=flat">
</a>

<a target="_blank" href="https://huggingface.co/TIGER-Lab/MAmmoTH-VL2">
<img style="height:22pt" src="https://img.shields.io/badge/-🤗%20Models-red?style=flat">
</a>

<a target="_blank" href="https://huggingface.co/spaces/TIGER-Lab/MAmmoTH-VL2">
<img style="height:22pt" src="https://img.shields.io/badge/-🤗%20Demo-red?style=flat">
</a>

<a target="_blank" href="https://tiger-ai-lab.github.io/VisualWebInstruct/">
<img style="height:22pt" src="https://img.shields.io/badge/-📝%20Website-red?style=flat">
</a>

<br>


## Overview
We utilize Google Search as a tool to augment multimodal reasoning dataset:
<img width="800" alt="abs" src="teaser.jpg">



## Introduction
Vision-Language Models have made significant progress on many perception-focused tasks, however, their progress on reasoning-focused tasks seem to be limited due to the lack of high-quality and diverse training data. In this work, we aim to address the scarcity issue of reasoning-focused multimodal datasets. We propose VisualWebInstruct - a novel approach that leverages search engine to create a diverse, and high-quality dataset spanning multiple disciplines like math, physics, finance, chemistry, etc. Starting with meticulously selected 30,000 seed images, we employ Google Image search to identify websites containing similar images. We collect and process the HTMLs from over 700K unique URL sources. Through a pipeline of content extraction, filtering and synthesis, we build a dataset of approximately 900K question-answer pairs, with 40% being visual QA pairs and the rest as text QA pairs. Models fine-tuned on VisualWebInstruct demonstrate significant performance gains: (1) training from Llava-OV-mid shows 10-20% absolute point gains across benchmarks, (2) training from MAmmoTH-VL shows 5% absoluate gain. Our best model MAmmoTH-VL2 shows state-of-the-art performance within the 10B parameter class on MMMU-Pro-std (40.7%), MathVerse (42.6%), and DynaMath (55.7%). These remarkable results highlight the effectiveness of our dataset in enhancing VLMs' reasoning capabilities for complex multimodal tasks.

## Repository Structure

The repository is organized into the following directories:

### VisualWebInstruct
Contains the data processing pipeline used to create the dataset:

- **Stage 1: Mining Data from the Internet**
  - Google Image searching
  - Accessibility tree building
  - QA pair extraction
  - Post-processing

- **Stage 2: Dataset Refinement**
  - Answer refinement with consistency checking
  - Answer alignment with original web content

### MAmmoTH-VL
Contains code for model training and evaluation. Since we finetune our model based on MAmmoTH-VL, we use the same codebase:

- **train**: Scripts for finetuning MAmmoTH-VL on VisualWebInstruct

- **evaluation**: Code for evaluating the model on various benchmarks

## Dataset Statistics

Our dataset exhibits the following distribution across knowledge domains:

| Category | Percentage |
|----------|------------|
| Math | 62.50% |
| Physics | 14.50% |
| Finance | 7.25% |
| Chemistry | 4.80% |
| Engineering | 4.35% |
| Others | 6.60% |

The "Others" category includes General Knowledge (2.45%), Computer Science (2.25%), Biology (1.40%), and humanities subjects.

## Model Performance

Models fine-tuned on VisualWebInstruct demonstrate significant performance gains:

1. Training from Llava-OV-mid shows 10-20% absolute point gains across benchmarks
2. Training from MAmmoTH-VL shows 5% absolute gain

Our best model MAmmoTH-VL2 shows state-of-the-art performance within the 10B parameter class on:
- MMMU-Pro-std (40.7%)
- MathVerse (42.6%)
- DynaMath (55.7%)


## Dataset Access

The VisualWebInstruct dataset is available on [Hugging Face](https://huggingface.co/datasets/TIGER-Lab/VisualWebInstruct).

To download the data for finetuning, you can use
```bash
# Data Preparation

export DATA_DIR=xxx #set your data folder first

huggingface-cli download TIGER-Lab/VisualWebInstruct --repo-type dataset --revision main --local-dir $DATA_DIR

unzip $DATA_DIR/images.zip -d $DATA_DIR/imgs
```
After unzipping, the folder structure in `$DATA_DIR/imgs` will be:

```
$DATA_DIR/imgs/
├── CLEVR_v1.0
├── ai2d
├── chartqa
├── coco
├── data
├── docvqa
├── geoqa+
├── gqa
├── llava
├── ocr_vqa
├── pisc
├── sam
├── share_textvqa
├── sqa
├── textvqa
├── vg
├── visualwebinstruct <-- This is the image folder of our dataset
├── web-celebrity
├── web-landmark
└── wikiart
```
## Model Training


### Environment Setup

```bash
# System configuration
export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=INFO

# Model configuration
export LLM_VERSION="Qwen/Qwen2.5-7B-Instruct"
export LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
export VISION_MODEL_VERSION="google/siglip-so400m-patch14-384"
export VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"
export PROMPT_VERSION="qwen_2_5"

# Path configuration
export HF_HOME=<your_huggingface_cache_path>
export IMAGE_FOLDER="$DATA_DIR/imgs"
export OUTPUT_DIR=<path_to_output_directory>

# Wandb configuration
export WANDB_API_KEY=<your_wandb_api_key>

# Training configuration
export BASE_RUN_NAME=<your_run_name>
export CKPT_PATH=$LLM_VERSION  # this could be the previous stage checkpoint like MammothVL
export NUM_GPUS=<number_of_gpus>
export NNODES=<number_of_nodes>
export RANK=<node_rank>
export ADDR=<master_address>
export PORT=<master_port>
export CUDA_VISIBLE_DEVICES=<gpu ids>

```

### Login to Weights & Biases

```bash
wandb login --relogin $WANDB_API_KEY
```

### Run Training
```bash
cd train/LLaVA-NeXT
```

You can run from commandline:
```bash
ACCELERATE_CPU_AFFINITY=1 torchrun --nproc_per_node="${NUM_GPUS}" --nnodes="${NNODES}" --node_rank="${RANK}" --master_addr="${ADDR}" --master_port="${PORT}" \
    llava/train/train_mem.py \
    --deepspeed scripts/zero3.json \
    --model_name_or_path ${CKPT_PATH} \
    --version ${PROMPT_VERSION} \
    --data_path scripts/train/mammoth_vl/visualwebinstruct.yaml \
    --image_folder ${IMAGE_FOLDER} \
    --video_folder "" \
    --mm_tunable_parts="mm_vision_tower,mm_mlp_adapter,mm_language_model" \
    --mm_vision_tower_lr=2e-6 \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --image_aspect_ratio anyres_max_4 \
    --image_grid_pinpoints  "(1x1),...,(6x6)" \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --run_name $BASE_RUN_NAME \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 20 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --torch_compile True \
    --torch_compile_backend "inductor" \
    --dataloader_drop_last True \
    --frames_upbound 32
```
Alternatively, you can also run the training script:
```bash
bash scripts/train/mammoth_vl/finetune_visualwebinstruct.sh
```
### Dataset Configuration

You'll need to set up the VisualWebInstruct dataset YAML file. Create or modify the YAML file at `VisualWebInstruct/MAmmoTH-VL/train/LLaVA-NeXT/scripts/train/mammoth_vl/visualwebinstruct.yaml` with the following content:

```yaml
datasets:
  - json_path:  # Path to the jsonl file of visualwebinstruct
    sampling_strategy: "all"
```

This configuration uses the `$DATA_DIR` environment variable that you set in the environment setup section.

### Notes

- This script trains a multimodal model combining Qwen2.5-7B-Instruct with SigLIP vision model
- The training uses DeepSpeed ZeRO-3 for optimization
- Parameters like `NUM_GPUS`, `NNODES`, etc. should be set according to your environment
- Replace placeholder values (indicated by `<...>`) with your actual configuration

## Evaluation

### Installation
```base
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv eval
uv venv --python 3.12
source eval/bin/activate
cd MAmmoTH-VL/train/LLaVA-NeXT/
uv pip install -e .
cd -
cd MAmmoTH-VL/eval/lmms-eval
uv pip install -e .
cd -
```

### Setup Environment
Enter the evaluation folder.

```bash
# Required environment variables
export HF_TOKEN=<your_huggingface_token>
export OPENAI_API_KEY=<your_openai_api_key>
export MODEL_PATH=TIGER-Lab/MAmmoTH-VL2
export TASK_NAME=mmmu_pro_standard
export OUTPUT_PATH=./log/

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=18000000
export NCCL_DEBUG=DEBUG
```

To evaluate the model:
```bash
CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes=1 \
    -m lmms_eval \
    --model llava_onevision \
    --model_args pretrained=${MODEL_PATH},conv_template=qwen_2_5,model_name=llava_qwen \
    --tasks ${TASK_NAME} \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix ${TASK_NAME} \
    --output_path ${OUTPUT_PATH}
```

## Pretrained Models

Our pretrained models are available on [Hugging Face](https://huggingface.co/TIGER-Lab/MAmmoTH-VL2).

## Acknowledgements

Our implementation builds upon the following codebases:
- [MAmmoTH-VL](https://github.com/MAmmoTH-VL/MAmmoTH-VL)
- [LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT)
- [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval)

We thank the authors of these repositories for their valuable contributions.

## Citation
```
@article{visualwebinstruct,
    title={VisualWebInstruct: Scaling up Multimodal Instruction Data through Web Search},
    author = {Jia, Yiming and Li, Jiachen and Yue, Xiang and Li, Bo and Nie, Ping and Zou, Kai and Chen, Wenhu},
    journal={arXiv preprint arXiv:2503.10582},
    year={2025}
}
```
