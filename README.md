# Transformer with Implicit Edges for Particle-based Physics Simulation (ECCV 2022)

\[[Paper](https://github.com/ftbabi/TIE_ECCV2022.git)\]

This is the official repository of "Transformer with Implicit Edges for Particle-based Physics Simulation, ECCV 2022". This repository contains *codes*, *pretrained models*, and *video demos* of our work.

**Authors**: Yidi Shao, [Chen Change Loy](https://www.mmlab-ntu.com/person/ccloy/),  and [Bo Dai](http://daibo.info/).

**Acknowedgement**: This study is supported under the RIE2020 Industry Alignment Fund Industry Collaboration Projects (IAF-ICP) Funding Initiative, as well as cash and in-kind contribution from the industry partner(s). It is also supported by Singapore MOE AcRF Tier 2 (MOE-T2EP20221-0011) and Shanghai AI Laboratory. 

**Feel free to ask questions. I am currently working on some other stuff but will try my best to reply. Please don't hesitate to star!** 

## News
- 19 July 2022: Training, test, and rendering codes released

## Table of Content
1. [Video Demos](#video-demos)
2. [Dataset](#dataset)
3. [Code](#code)
4. [Citations](#citations)

## Video Demos
Coming soon.

## Dataset
Please follow [this repo](https://github.com/YunzhuLi/DPI-Net.git) to generate your own dataset.

## Code
Codes are tested on Ubuntu 18 and cuda 9.2.

### Installation
1. Create a conda environment
```
conda create -n TIE python=3.6
conda activate TIE
```
2. Clone and install this repo
```
git clone https://github.com/ftbabi/TIE_ECCV2022.git

cd TIE_ECCV2022
pip install -v -e .
```
3. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/get-started/locally/), e.g.,
```
conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=9.2 -c pytorch
```
4. Install mmcv-full
```
pip install mmcv-full==1.3.1 -f https://download.openmmlab.com/mmcv/dist/cu92/torch1.7.0/index.html
```
5. Install dependent packages

```
pip install h5py==2.8.0 scipy==1.5.0 tqdm scikit-learn==0.23.1

conda install pybind11==2.7.0  # This is for rendering
```

### Training
1. Train on slurm system
```
./tools/slurm_train.sh {PARTITION} {JOB_NAME} {CONFIG} {WORK_DIR} # please refer to `tools/slurm_train.sh for more details
```
2. Train on multiple GPUs
```
python tools/train.py {CONFIG} {WORK_DIR}
```

### Testing
1. Predict rollout on slurm system
```
./tools/slurm_predict.sh {PARTITION} {JOB_NAME} {CONFIG} {WORK_DIR} --checkpoint {CHECKPOINT}
```
2. Predict rollout on single GPU
```
python tools/predict_rollout.py {CONFIG} {WORK_DIR} --checkpoint {CHECKPOINT}
```

### Rendering
Please make sure you have followed [this repo](https://github.com/YunzhuLi/DPI-Net.git) to generate your own dataset.
```
python tools/render_rollout.py {CONFIG} {SRC_DIR} --save_dir {SAVE_DIR}
```
Please refer to `tools/render_rollout.py` for more details.

## Citations
```
@inproceedings{shao2022transformer,
  author = {Shao, Yidi and Loy, Chen Change and Dai, Bo},
  title = {Transformer with Implicit Edges for Particle-based Physics Simulation},
  booktitle = {Computer Vision - {ECCV} 2022 - 17th European Conference},
  year = {2022}
}
```
