import torch
import os
import numpy as np
import sys
sys.path.append('/root/mwnoh/smoothquant/smoothquant')

from opt import Int8OPTForCausalLM

model_smoothquant = Int8OPTForCausalLM.from_pretrained(
    'mit-han-lab/opt-125m-smoothquant', torch_dtype=torch.float16, device_map='auto')

output_dir = "opt-125m-weights-pt"
os.makedirs(output_dir, exist_ok=True)
state_dict = model_smoothquant.state_dict()
for name, weight_tensor in state_dict.items():
    name += '.pt'
    filename = os.path.join(output_dir, name)
    if weight_tensor.dtype != torch.int8:
        weight_tensor = weight_tensor.to(torch.float32).cpu()
    elif weight_tensor.dtype == torch.int8:
        weight_tensor = weight_tensor.to(torch.int32).cpu()
    torch.save(weight_tensor, filename)
    print(f"Saved {name} -> {filename}")