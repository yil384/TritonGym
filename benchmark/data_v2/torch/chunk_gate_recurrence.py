import torch

class ChunkGateRecurrent(torch.autograd.Function):
    @staticmethod
    def forward(ctx, kv, cross_decay, last_kv=None):
        # kv: [B, H, N, D_k, D_v]
        # cross_decay: [B, H, N]
        # last_kv: [B, H, D_k, D_v] or None
        B, H, N, D_k, D_v = kv.shape
        output = torch.empty_like(kv)
        BLOCK_MODEL_K = 64
        BLOCK_MODEL_V = 16
        assert D_k % BLOCK_MODEL_K == 0
        assert D_v % BLOCK_MODEL_V == 0
        # Forward recurrence
        for b in range(B):
            for h_ in range(H):
                acc = last_kv[b, h_] if last_kv is not None else torch.zeros(D_k, D_v, device=kv.device, dtype=kv.dtype)
                for n in range(N):
                    output[b, h_, n] = acc
                    acc = acc * cross_decay[b, h_, n] + kv[b, h_, n]
        ctx.save_for_backward(output, cross_decay)
        ctx.have_last_kv = last_kv is not None
        return output

    @staticmethod
    def backward(ctx, DO):
        output, cross_decay = ctx.saved_tensors
        B, H, N, D_k, D_v = output.shape
        DI = torch.empty_like(DO)
        DG = torch.empty(B, H, N, device=cross_decay.device, dtype=cross_decay.dtype)
        DL = torch.empty(B, H, D_k, D_v, device=output.device, dtype=output.dtype)
        # Backward recurrence
        for b in range(B):
            for h_ in range(H):
                Dacc = torch.zeros(D_k, D_v, device=output.device, dtype=output.dtype)
                for n in reversed(range(N)):
                    Dacc = Dacc * cross_decay[b, h_, n] + DO[b, h_, n]
                    DI[b, h_, n] = Dacc
                    DG[b, h_, n] = (Dacc * output[b, h_, n]).sum()
                DL[b, h_] = Dacc
        # Zero out the gradient of the last block
        DI[:, :, -1] = 0
        DG[:, -1] = 0
        return DI, DG, DL if ctx.have_last_kv else None

torch_chunk_gate_recurrence = ChunkGateRecurrent.apply
