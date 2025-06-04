import torch
import os
import sys
from transformers import AutoTokenizer
from torch.nn.functional import pad
import pickle
sys.path.append('/root/mwnoh/smoothquant/smoothquant')
from opt import Int8OPTForCausalLM

DEVICE = torch.device("cuda")  

def generate(model, input_ids, past_key_values=None):
    model.eval()
    if past_key_values is None:
        pad_len = 512 - input_ids.shape[1]
        input_ids = pad(input_ids, (0, pad_len), value=1)
        torch.cuda.synchronize()
        outputs = model(input_ids)
        torch.cuda.synchronize()
    else:
        torch.cuda.synchronize()
        outputs = model(input_ids=input_ids, past_key_values=past_key_values)
        torch.cuda.synchronize()
    
    return outputs

opt_model = "opt-125m"

model_smoothquant = Int8OPTForCausalLM.from_pretrained(
    f'/mnt/models/smoothquant/{opt_model}', torch_dtype=torch.float16, device_map='auto')

tokenizer = AutoTokenizer.from_pretrained('facebook/opt-125m')
input = "Earth rotates around the sun, and moon rotates around the "
input_ids_list = tokenizer(input).input_ids
input_ids = torch.tensor(input_ids_list).unsqueeze(0).cuda(DEVICE)

outputs = generate(model_smoothquant, input_ids)
past_key_values = outputs.past_key_values

kvs_output_dir = f"/root/mwnoh/smoothquant/examples/kvs/{opt_model}"
os.makedirs(kvs_output_dir, exist_ok=True)
for i, past_key_value in enumerate(past_key_values):
    past_keys_filename = f"{i}.past_keys.pt"
    past_keys_filename = os.path.join(kvs_output_dir, past_keys_filename)
    _, _, seq_len, head_dim = past_key_value[0].shape
    past_keys = past_key_value[0]
    past_keys = past_keys.view(-1, seq_len, head_dim)
    past_keys = past_keys.to(torch.int32).cpu()
    past_values_filename = f"{i}.past_values.pt"
    past_values_filename = os.path.join(kvs_output_dir, past_values_filename)
    past_values = past_key_value[1]
    past_values = past_values.view(-1, seq_len, head_dim)
    past_values = past_values.to(torch.int32).cpu()
    torch.save(past_keys, past_keys_filename)
    torch.save(past_values, past_values_filename)

with open("past_key_values.pkl", "wb") as f:
    pickle.dump(past_key_values, f)

next_token_logits = outputs.logits[:, -1, :]
next_token_id = torch.argmax(next_token_logits, dim=-1)  # [batch_size]
next_token_id = next_token_id.unsqueeze(-1)
next_token_id_cpu = next_token_id.cpu()
next_token_id_output_dir = f"/root/mwnoh/smoothquant/examples/next_token_ids/{opt_model}"
os.makedirs(next_token_id_output_dir, exist_ok=True)
filename = "next_token_id.pt"
filename = os.path.join(next_token_id_output_dir, filename)
torch.save(next_token_id_cpu, filename)


# outputs = generate(model_smoothquant, next_token_id, past_key_values)


