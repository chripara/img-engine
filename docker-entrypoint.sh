#!/bin/bash
set -e

mkdir -p local_models

if [ ! -f local_models/RealESRGAN_x4plus.pth ]; then
    echo "Downloading RealESRGAN_x4plus.pth..."
    curl -L -o local_models/RealESRGAN_x4plus.pth \
        https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth
fi

if [ ! -f local_models/RealESRGAN_x4plus_anime_6B.pth ]; then
    echo "Downloading RealESRGAN_x4plus_anime_6B.pth..."
    curl -L -o local_models/RealESRGAN_x4plus_anime_6B.pth \
        https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth
fi

exec python run.py
