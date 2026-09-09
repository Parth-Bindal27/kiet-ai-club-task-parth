"""Hardware detection and recommendation utility."""
import platform
import psutil
from dataclasses import dataclass
try:
    import torch
except ImportError:
    torch = None

@dataclass
class HardwareInfo:
    cpu_count: int
    ram_gb: float
    gpu_type: str
    gpu_count: int
    vram_gb: float
    recommended_model_size: str
    recommended_batch_size: int

def get_hardware_info() -> HardwareInfo:
    cpu_count = psutil.cpu_count(logical=True) or 1
    ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    
    gpu_type = "None"
    gpu_count = 0
    vram_gb = 0.0
    
    if torch is not None:
        if torch.cuda.is_available():
            gpu_type = "CUDA"
            gpu_count = torch.cuda.device_count()
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            gpu_type = "MPS"
            gpu_count = 1
            vram_gb = ram_gb * 0.7  # MPS uses shared memory
            
    if vram_gb >= 24:
        rec_model = "7B-13B"
        rec_batch = 16
    elif vram_gb >= 12:
        rec_model = "7B"
        rec_batch = 8
    elif vram_gb > 0:
        rec_model = "1.5B-3B"
        rec_batch = 4
    else:
        rec_model = "small/base"
        rec_batch = 2
        
    return HardwareInfo(
        cpu_count=cpu_count,
        ram_gb=ram_gb,
        gpu_type=gpu_type,
        gpu_count=gpu_count,
        vram_gb=vram_gb,
        recommended_model_size=rec_model,
        recommended_batch_size=rec_batch
    )
