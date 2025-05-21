from torch_int.nn.bmm import BMM_S8T_S8N_F32T
import torch
import numpy as np
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

q_tensor = torch.load("/root/mwnoh/smoothquant/examples/my_debug_outputs/query_states.pt")   # shape [B, M, K], dtype=int8
k_tensor = torch.load("/root/mwnoh/smoothquant/examples/my_debug_outputs/key_states.pt")     # shape [B, N, K], dtype=int8
a_tensor = torch.load("/root/mwnoh/smoothquant/examples/my_debug_outputs/attn_weights.pt")

q_tensor_int32 = q_tensor.to(torch.int32)
k_tensor_int32 = k_tensor.to(torch.int32)

print(f"q_tensor_int32 dtype: {q_tensor_int32.dtype}")
print(f"q_tensor_int32 dtype: {k_tensor_int32.dtype}")
print(f"q_tensor_int32 dtype: {a_tensor.dtype}")


# print("Shapes before BMM:") 
# print("q_tensor:", q_tensor.shape)
# print("k_tensor:", k_tensor.shape)

q_vector = q_tensor[0, 0, :].unsqueeze(0).unsqueeze(0)
k_matrix = k_tensor[0, :, :].unsqueeze(0)

q_vector_int32 = q_tensor_int32[0, 0, :].unsqueeze(0).unsqueeze(0)
k_matrix_int32 = k_tensor_int32[0, :, :].unsqueeze(0)


alpha = 0.0006
alpha_torch = torch.tensor(alpha, dtype=torch.float32)
qk_bmm = BMM_S8T_S8N_F32T(alpha)
qk_bmm.a = alpha_torch

# print("q_vector: \n", q_vector)
# print("k_matrix: \n", k_matrix)
result = qk_bmm(q_vector, k_matrix)
# print("result: \n", result[0, 0, :])


result_num = result.cpu().numpy()
q_vector_num = q_vector_int32.cpu().numpy()
k_matrix_num = k_matrix_int32.cpu().numpy()
k_matrix_num_t = np.swapaxes(k_matrix_num, 1, 2)

print("q_vector_num dtype: ", q_vector_num.dtype)
print("k_matrix_num dtype: ", k_matrix_num.dtype)

a_interm = np.matmul(q_vector_num, k_matrix_num_t)
print("a_interm dtype: ", a_interm.dtype)
# print("a_interm: \n", a_interm[0, 0, :])
alpha_num = np.float32(alpha)
print("alpha_num dtype: ", alpha_num.dtype)
a_final = alpha_num * a_interm
print("a_final dtype: ", a_final.dtype)
# print("a_final: \n", a_final[0, 0, :])

is_close = np.allclose(a_final, result_num, atol=1e-2, rtol=1e-2)
print(f"Match within tolerance?: {is_close}")

os.makedirs("/root/mwnoh/smoothquant/examples/qk_bmm_in_out", exist_ok=True)
np.save("/root/mwnoh/smoothquant/examples/qk_bmm_in_out/query_vector.npy", q_vector_num)
np.save("/root/mwnoh/smoothquant/examples/qk_bmm_in_out/key_matrix.npy", k_matrix_num)
np.save("/root/mwnoh/smoothquant/examples/qk_bmm_in_out/a_interm.npy", a_interm)
np.save("/root/mwnoh/smoothquant/examples/qk_bmm_in_out/alpha_num.npy", alpha_num)
np.save("/root/mwnoh/smoothquant/examples/qk_bmm_in_out/a_final.npy", a_final)