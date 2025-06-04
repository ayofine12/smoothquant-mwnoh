import torch

attn_probs = torch.load("./my_debug_outputs/attn_probs.pt").cpu().to(torch.int32)
value_states = torch.load("./my_debug_outputs/value_states.pt").cpu().to(torch.int32)

print("attn_probs: \n", attn_probs[0])
print("value_states: \n", value_states[0][0])

interm = torch.einsum('h d s, h s -> h d', value_states, attn_probs)
print("interm: \n", interm[0][0])
torch.save(interm, 'interm.pt')
print("interm shape: ", interm.shape)