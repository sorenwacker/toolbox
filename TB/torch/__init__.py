import torch
from .AutoEncoder import AutoEncoder
from .Transformer import Transformer

print(f'CUDA is available: {torch.cuda.is_available()}')
print(f'CUDA current device: {torch.cuda.current_device()} ({torch.cuda.get_device_name()})')
    