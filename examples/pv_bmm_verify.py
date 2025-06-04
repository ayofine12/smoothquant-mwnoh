from torch_int.nn.bmm import BMM_S8T_S8N_S8T
import torch
import numpy as np
import os

attn_probs = torch.load("/root/mwnoh/smoothquant/examples/my_debug_outputs/attn_probs.pt")
value_states = torch.load("/root/mwnoh/smoothquant/examples/my_debug_outputs/value_states.pt")
pv_bmm_alpha = torch.load("/root/mwnoh/smoothquant/examples/opt-125m-np/model.decoder.layers.0.self_attn.pv_bmm.a.pt")
attn_output = torch.load("/root/mwnoh/smoothquant/examples/my_debug_outputs/attn_output.pt")
print("attn_output: \n", attn_output[0, 0, :])

print("attn_probs shape: ", attn_probs.shape)
print("value_states shape: ", value_states.shape)
print("pv_bmm_alpha shape: ", pv_bmm_alpha.shape)
print("attn_output shape: ", attn_output.shape)

attn_probs_float32 = attn_probs.to(torch.float32)
value_states_float32 = value_states.to(torch.float32)
pv_bmm_alpha_float32 = pv_bmm_alpha.to(torch.float32)
attn_output_float32 = attn_output.to(torch.float32)

pv_bmm = BMM_S8T_S8N_S8T(1.0)
pv_bmm.a = pv_bmm_alpha

result = pv_bmm(attn_probs, value_states)
print("result: \n", result[0, 0, :])

value_states_float32_t = value_states_float32.transpose(1, 2)

interm = torch.bmm(attn_probs_float32, value_states_float32_t)

interm *= pv_bmm_alpha_float32

interm_rounded = torch.round(interm)

out = torch.clamp(interm_rounded, min=-128, max=127).to(torch.int8)


# if torch.equal(out, attn_output):
#     print("output_tensor and result are exactly the same.")
# else:
#     print("output_tensor and result are NOT exactly the same.")
#     max_abs_diff = (out - attn_output).abs().max()
#     print("Maximum absolute difference:", max_abs_diff.item())
