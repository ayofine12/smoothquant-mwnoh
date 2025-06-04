from torch_int.nn.linear import W8A8B8O8Linear
import torch

hidden_states = torch.load("/root/mwnoh/smoothquant/examples/my_debug_outputs/hidden_states.pt")
query_states = torch.load("/root/mwnoh/smoothquant/examples/my_debug_outputs/query_states.pt")
print("query_states: \n", query_states)

q_proj_weight = torch.load("/root/mwnoh/smoothquant/examples/opt-125m-weights-pt/model.decoder.layers.0.self_attn.q_proj.weight.pt")
q_proj_bias = torch.load("/root/mwnoh/smoothquant/examples/opt-125m-weights-pt/model.decoder.layers.0.self_attn.q_proj.bias.pt")
q_proj_a = torch.load("/root/mwnoh/smoothquant/examples/opt-125m-weights-pt/model.decoder.layers.0.self_attn.q_proj.a.pt")
q_proj_b = torch.load("/root/mwnoh/smoothquant/examples/opt-125m-weights-pt/model.decoder.layers.0.self_attn.q_proj.b.pt")

hidden_states_float32 = hidden_states.to(torch.float32)
hiddens_states_shape = hidden_states_float32.shape
hidden_states_float32 = hidden_states_float32.view(-1, hiddens_states_shape[-1])
q_proj_weight_float32 = q_proj_weight.to(torch.float32)
q_proj_bias_float32 = q_proj_bias.to(torch.float32)
q_proj_a_float32 = q_proj_a.to(torch.float32)
q_proj_b_float32 = q_proj_b.to(torch.float32)

print("hidden_states_float32 shape: ", hidden_states_float32.shape)
print("q_proj_weight_float32 shape: ", q_proj_weight_float32.shape)
print("q_proj_bias_float32 shape: ", q_proj_bias_float32.shape)
print("q_proj_a_float32 shape: ", q_proj_a_float32.shape)
print("q_proj_b_float32 shape: ", q_proj_b_float32.shape)

interm = hidden_states_float32.mm(q_proj_weight_float32.t())
interm = q_proj_a_float32 * interm + q_proj_b_float32 * q_proj_bias_float32

interm_rounded = torch.round(interm)
out = torch.clamp(interm_rounded, min=-128, max=127).to(torch.int8)
out = out.view(*hiddens_states_shape[:-1], -1)
print("out: \n", out)

if torch.equal(out, query_states):
    print("output_tensor and result are exactly the same.")
else:
    print("output_tensor and result are NOT exactly the same.")
    max_abs_diff = (out - query_states).abs().max()
    print("Maximum absolute difference:", max_abs_diff.item())