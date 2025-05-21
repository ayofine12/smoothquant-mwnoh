from torch_int.nn.linear import W8A8B8O8LinearReLU
import torch
import numpy as np
import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '2'

input_tensor = torch.load("/root/mwnoh/smoothquant/examples/my_debug_outputs/f1_input.pt")
embed_dim = input_tensor.size(2)
output_tensor = torch.load("/root/mwnoh/smoothquant/examples/my_debug_outputs/f1_output.pt")
print("output_tensor: \n", output_tensor)
ffn_dim = output_tensor.size(2)

f1_weight_tensor = torch.load("/root/mwnoh/smoothquant/examples/opt-125m-np/model.decoder.layers.0.fc1.weight.pt")
f1_weight_tensor = f1_weight_tensor.to('cuda:0')
f1_bias_tensor = torch.load("/root/mwnoh/smoothquant/examples/opt-125m-np/model.decoder.layers.0.fc1.bias.pt")
f1_bias_tensor = f1_bias_tensor.to('cuda:0')
f1_alpha_tensor = torch.load("/root/mwnoh/smoothquant/examples/opt-125m-np/model.decoder.layers.0.fc1.a.pt")
f1_alpha_tensor = f1_alpha_tensor.to('cuda:0')
f1_beta_tensor = torch.load("/root/mwnoh/smoothquant/examples/opt-125m-np/model.decoder.layers.0.fc1.b.pt")
f1_beta_tensor = f1_beta_tensor.to('cuda:0')

input_tensor_int32 = input_tensor.to(torch.int32)
output_tensor_int32 = output_tensor.to(torch.int32)

f1_weight_tensor_int32 = f1_weight_tensor.to(torch.int32)
f1_bias_tensor_int32 = f1_bias_tensor.to(torch.int32)
f1_alpha_tensor_int32 = f1_alpha_tensor.to(torch.float32)
f1_beta_tensor_int32 = f1_beta_tensor.to(torch.float32)

print("input_tensor_int32 shape: ", input_tensor_int32.shape)
print("output_tensor_int32 shape: ", output_tensor_int32.shape)

print("f1_weight_tensor_int32 shape: ", f1_weight_tensor_int32.shape)
print("f1_bias_tensor_int32 shape: ", f1_bias_tensor_int32.shape)
print("f1_alpha_tensor_int32 shape: ", f1_alpha_tensor_int32.shape)
print("f1_beta_tensor_int32 shape: ", f1_beta_tensor_int32.shape)

print("f1_alpha_tensor_int32: \n", f1_alpha_tensor_int32)
print("f1_beta_tensor_int32: \n", f1_beta_tensor_int32)

input_vector = input_tensor[0, 0, :].unsqueeze(0).unsqueeze(0)

fc1 = W8A8B8O8LinearReLU(embed_dim, ffn_dim)

fc1.weight = f1_weight_tensor
print("fc1.weight: ", fc1.weight)
fc1.bias = f1_bias_tensor
fc1.a = f1_alpha_tensor
print("fc1.a: ", fc1.a)
fc1.b = f1_beta_tensor

result = fc1(input_tensor)
print("result: \n", result)

if torch.equal(output_tensor, result):
    print("output_tensor and result are exactly the same.")
else:
    print("output_tensor and result are NOT exactly the same.")
    max_abs_diff = (output_tensor - result).abs().max()
    print("Maximum absolute difference:", max_abs_diff.item())





# print("Shapes before BMM:") 
# print("q_tensor:", q_tensor.shape)
# print("k_tensor:", k_tensor.shape)

# q_vector = q_tensor[0, 0, :].unsqueeze(0).unsqueeze(0)
# k_matrix = k_tensor[0, :, :].unsqueeze(0)

# q_vector_int32 = q_tensor_int32[0, 0, :].unsqueeze(0).unsqueeze(0)
# k_matrix_int32 = k_tensor_int32[0, :, :].unsqueeze(0)


# alpha = 0.0006
# alpha_torch = torch.tensor(alpha, dtype=torch.float32)
# qk_bmm = BMM_S8T_S8N_F32T(alpha)
# qk_bmm.a = alpha_torch

# # print("q_vector: \n", q_vector)
# # print("k_matrix: \n", k_matrix)
# result = qk_bmm(q_vector, k_matrix)
# # print("result: \n", result[0, 0, :])


# result_num = result.cpu().numpy()
# q_vector_num = q_vector_int32.cpu().numpy()
# k_matrix_num = k_matrix_int32.cpu().numpy()
# k_matrix_num_t = np.swapaxes(k_matrix_num, 1, 2)

# print("q_vector_num dtype: ", q_vector_num.dtype)
# print("k_matrix_num dtype: ", k_matrix_num.dtype)

# a_interm = np.matmul(q_vector_num, k_matrix_num_t)
# print("a_interm dtype: ", a_interm.dtype)
# # print("a_interm: \n", a_interm[0, 0, :])
# alpha_num = np.float32(alpha)
# print("alpha_num dtype: ", alpha_num.dtype)
# a_final = alpha_num * a_interm
# print("a_final dtype: ", a_final.dtype)
# # print("a_final: \n", a_final[0, 0, :])

# is_close = np.allclose(a_final, result_num, atol=1e-2, rtol=1e-2)
# print(f"Match within tolerance?: {is_close}")

# os.makedirs("/root/mwnoh/smoothquant/examples/qk_bmm_in_out", exist_ok=True)
# np.save("/root/mwnoh/smoothquant/examples/qk_bmm_in_out/query_vector.npy", q_vector_num)
# np.save("/root/mwnoh/smoothquant/examples/qk_bmm_in_out/key_matrix.npy", k_matrix_num)
# np.save("/root/mwnoh/smoothquant/examples/qk_bmm_in_out/a_interm.npy", a_interm)
# np.save("/root/mwnoh/smoothquant/examples/qk_bmm_in_out/alpha_num.npy", alpha_num)
# np.save("/root/mwnoh/smoothquant/examples/qk_bmm_in_out/a_final.npy", a_final)