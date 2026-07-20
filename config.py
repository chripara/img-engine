import os
from utils.enums.checkpoint import Checkpoint

class Config:
    DEBUG = False
    CHECKPOINT = Checkpoint.SDXL_BASE  # Checkpoint Selector

class DevelopmentConfig(Config):
    DEBUG = True