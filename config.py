import os
from utils.enums import ImgBackend

class Config:
    DEBUG = False
    IMG_BACKEND = ImgBackend.SDXL  # Backend Selector

class DevelopmentConfig(Config):
    DEBUG = True