import os
import torch
import numpy as np
from pathlib import Path

input_dir = "opt-125m-weights-pt"
output_dir = "opt-125m-weights-npy"
os.makedirs(output_dir, exist_ok=True)

file_num_total = 0
file_num_count = 0

for filename in os.listdir(input_dir):
    full_input_path = os.path.join(input_dir, filename)
    weight_tensor = torch.load(full_input_path)

    filename = Path(filename).stem
    full_output_path = os.path.join(output_dir, filename)
    full_output_path += ".npy"
    weight_npy = weight_tensor.cpu().numpy()
    if weight_tensor.dtype != torch.int8:
        weight_npy = weight_npy.astype(np.float32)
        file_num_count += 1
    elif weight_tensor.dtype == torch.int8:
        weight_npy = weight_npy.astype(np.int32)
        file_num_count += 1
    np.save(full_output_path, weight_npy)
    file_num_total += 1

print("file_num_total == file_num_count -->", file_num_total == file_num_count)
        
        
        