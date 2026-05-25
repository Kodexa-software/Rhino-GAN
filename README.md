# RhinoGAN

![Hero Image](./images/samples.jpg)

## Framework Overview

![Overview](./images/work-overflow.jpg)

## Introduction

RhinoGAN is a surgeon-guided nasal editing system built on the FS latent space of StyleGAN2.  
It enables realistic nose structure transfer (NSB) and precise shape refinement (NSR) using segmentation masks and landmarks—all while preserving facial identity.  
Designed for clinical visualization, RhinoGAN received surgeon evaluation scores averaging 7.5/10 for realism and clinical usefulness.  
This repository provides a fully functional Docker-based demo for research and experimentation.

## [Demo Video](https://www.youtube.com/watch?v=NBWAY4dVwSM)

[![Watch the Video](https://img.youtube.com/vi/NBWAY4dVwSM/maxresdefault.jpg)](https://www.youtube.com/watch?v=NBWAY4dVwSM)

## Quick Start (Docker Setup)

Requirements:

- NVIDIA GPU
- Docker installed
- Docker daemon running

### 1. Clone the Project

```bash
git clone https://github.com/kodo-yousif/RhinoGAN
cd RhinoGAN
```

### 2. Download Pretrained FFHQ Model

Place into `backend/pretrained_models/`:
https://drive.google.com/file/d/1AT6bNR2ppK8f2ETL_evT27f3R_oyWNHS/view?usp=sharing

### 3. Build Docker Image (Powershell in admin mode)

```bash
docker build -t rhinogan-image .
```

### 4. Run Container (Powershell in admin mode)

```bash
docker run -d --gpus all --privileged -it -p 8000:3000 -p 8001:3001 -v .:/nose-ai --name rhinogan-container rhinogan-image
```

### 5. Access Container

```bash
docker exec -it rhinogan-container bash
```

### Optional Conda Fix

```bash
/opt/conda/bin/conda init bash
conda activate nose-ai
echo -e "source /opt/conda/etc/profile.d/conda.sh
conda activate nose-ai" >> ~/.bashrc
exit
```

### 6. Run backend and front end servers at different terminals at same time

```bash
cd frontend
yarn
yarn dev
```

```bash
cd backend
python main.py
```

Please manually run the backend and frontend folder. You can watch the demo video for full tutorial.

## 📧 Contact

kodo.yousif@gmail.com
