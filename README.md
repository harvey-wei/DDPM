# DDPM for Image Generation

The project structure is as follows:

```
ddpm_project/
│
├── configs/                  # YAML/JSON configs
│   └── default.yaml
│
├── models/                   # Network architectures
│   ├── unet.py
│   ├── resblock.py
│   ├── attention.py
│   ├── time_embedding.py
│   └── __init__.py
│
├── diffusion/                # DDPM forward/reverse processes
│   ├── diffusion.py
│   ├── scheduler.py
│   └── __init__.py
│
├── data/                     # Dataset loaders
│   ├── loader.py
│   └── transforms.py
│
├── metrics/                  # FID computation and Inception model
│   ├── inception.py          # Pretrained InceptionV3 wrapper
│   ├── measure_fid.py        # FID computation logic
│   └── __init__.py
│
├── utils/                    # General utilities
│   ├── utils.py
│   ├── ema.py
│   └── __init__.py
│
├── train.py                  # Training script
├── sample.py                 # Sampling/generated image script
├── evaluate.py               # FID/IS/LS evaluation
├── main.py                   # Central experiment runner (optional CLI)
│
├── requirements.txt
└── README.md

```
