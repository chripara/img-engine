import os
from utils.enums import ImgBackend
from 

class Config:
    DEBUG = False
    IMG_BACKEND = ImgBackend.SDXL  # Backend Selector

class DevelopmentConfig(Config):
    DEBUG = True