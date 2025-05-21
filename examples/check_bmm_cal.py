import numpy as np

def simulate_bmm_s8t_s8n_f32t_optionA_with_alpha():
    # 1) Load data
    query_states_cpu = np.load("my_debug_outputs/query_states_cpu.npy")   # shape [B, M, K], dtype=int8
    key_states_cpu   = np.load("my_debug_outputs/key_states_cpu.npy")     # shape [B, N, K], dtype=int8
    attn_weights_ref = np.load("my_debug_outputs/attn_weights_vector.npy")# shape [B, M, N], dtype=float32

    # 2) Shapes and checks
    B, M, Kq = query_states_cpu.shape
    B2, N, Kk = key_states_cpu.shape
    assert B == B2,    f"Batch size mismatch in Query({B}) vs Key({B2})"
    assert Kq == Kk,   f"K-dim mismatch in Query({Kq}) vs Key({Kk})"

    # 3) Transpose the key array to match column-major layout in CUTLASS
    #    key_states_cpu is shape [B, N, K], row-major in NumPy
    #    but bmm_s8t_s8n_f32t interprets it as [B, K, N] (column-major).
    key_states_cpu_t = np.swapaxes(key_states_cpu, -1, -2)  # shape [B, K, N]

    # 4) Set alpha (the scale factor in the CUTLASS code)
    alpha = 0.0006

    # 5) Perform the batched matmul in Python
    #    query [B, M, K], key_t [B, K, N] => result [B, M, N]
    #    We accumulate in int32, then multiply by alpha => float
    #    Using einsum: "bmk, bkn -> bmn"
    attn_weights_computed = alpha * np.einsum("bmk,bkn->bmn", query_states_cpu, key_states_cpu_t)

    # Cast to float32 if you want the exact type match
    attn_weights_computed = attn_weights_computed.astype(np.float32)

    # 6) Compare shapes and values
    print(f"[INFO] attn_weights_ref.shape = {attn_weights_ref.shape}")
    print(f"[INFO] attn_weights_computed.shape = {attn_weights_computed.shape}")

    # 7) Compare numerically
    max_abs_diff  = np.amax(np.abs(attn_weights_ref - attn_weights_computed))
    max_rel_diff  = np.amax(
        np.abs(attn_weights_ref - attn_weights_computed) 
        / np.maximum(1e-9, np.abs(attn_weights_ref))
    )
    is_close = np.allclose(attn_weights_ref, attn_weights_computed, atol=1e-2, rtol=1e-2)

    # 8) Print results
    print(f"Max absolute difference: {max_abs_diff}")
    print(f"Max relative difference: {max_rel_diff}")
    print(f"Match within tolerance?: {is_close}")

if __name__ == "__main__":
    simulate_bmm_s8t_s8n_f32t_optionA_with_alpha()