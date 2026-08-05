import os
from utils.enums.checkpoint import Checkpoint

class Config:
    DEBUG = False

class DevelopmentConfig(Config):
    DEBUG = True