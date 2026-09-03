# app/services/gpu_lock.py
import threading

_gpu_lock = threading.Semaphore(1)

def get_gpu_lock() -> threading.Semaphore:
    return _gpu_lock