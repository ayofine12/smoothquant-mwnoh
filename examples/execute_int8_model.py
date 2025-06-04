import sys
import pickle
import torch
sys.path.append('/root/mwnoh/smoothquant/smoothquant')
from opt import Int8OPTForCausalLM

opt_model = "opt-125m"

model = Int8OPTForCausalLM.from_pretrained(
    f'/mnt/models/smoothquant/{opt_model}', torch_dtype=torch.float16, device_map='auto')

next_token_id = torch.load(f'/root/mwnoh/smoothquant/examples/next_token_ids/{opt_model}/next_token_id.pt')
next_token_id = next_token_id.cuda(0)

with open('/root/mwnoh/smoothquant/examples/past_key_values.pkl', 'rb') as f:
    past_key_values = pickle.load(f)


outputs = model(input_ids=next_token_id, past_key_values=past_key_values)

