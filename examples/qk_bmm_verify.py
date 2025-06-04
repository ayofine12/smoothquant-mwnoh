from torch_int.nn.bmm import BMM_S8T_S8N_F32T
import torch
import numpy as np
import os

os.makedirs("/root/mwnoh/smoothquant/examples/my_debug_outputs", exist_ok=True)
q_tensor = torch.load("/root/mwnoh/smoothquant/examples/my_debug_outputs/query_states.pt")   # shape [B, M, K], dtype=int8
k_tensor = torch.load("/root/mwnoh/smoothquant/examples/my_debug_outputs/key_states.pt")     # shape [B, N, K], dtype=int8
a_tensor = torch.load("/root/mwnoh/smoothquant/examples/my_debug_outputs/attn_weights.pt")

q_tensor_int32 = q_tensor.to(torch.int32)
print("q_tensor_int32 shape: ", q_tensor_int32.shape)
k_tensor_int32 = k_tensor.to(torch.int32)
print("k_tensor_int32 shape: ", k_tensor_int32.shape)


a_accum = torch.einsum("hsd,hd->hs", k_tensor_int32, q_tensor_int32)
print("a_accum shape: ", a_accum.shape)
print("a_accum[0]: \n", a_accum[0])

# a_final = a_accum * alpha_tensor
# # print("a_final: \n", a_final)

# a_answer = a_tensor[0][0]

# if torch.equal(a_answer, a_final):
#     print("output_tensor and result are exactly the same.")
# else:
#     print("output_tensor and result are NOT exactly the same.")
#     max_abs_diff = (a_answer - a_final).abs().max()
#     print("Maximum absolute difference:", max_abs_diff.item())