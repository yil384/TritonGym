import torch

class TorchLightningAttention2NoDecay(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v):
        # q: [b, h, n, d], k: [b, h, n, d], v: [b, h, n, e]
        b, h, n, d = q.shape
        e = v.shape[-1]
        BLOCK = 64

        ctx.save_for_backward(q, k, v)

        o = torch.zeros(b, h, n, e, dtype=q.dtype, device=q.device)

        # Process in blocks, matching the Triton kernel logic
        NUM_BLOCK = (n + BLOCK - 1) // BLOCK
        for bh_idx in range(b * h):
            bi = bh_idx // h
            hi = bh_idx % h

            kv = torch.zeros(d, e, dtype=torch.float32, device=q.device)

            for i in range(NUM_BLOCK):
                start = i * BLOCK
                end = min(start + BLOCK, n)
                block_len = end - start

                q_block = q[bi, hi, start:end].float()  # [block_len, d]
                k_block = k[bi, hi, start:end].float()  # [block_len, d]
                v_block = v[bi, hi, start:end].float()  # [block_len, e]

                # Intra-block: causal attention (lower triangular)
                qk = torch.matmul(q_block, k_block.T)  # [block_len, block_len]
                causal_mask = torch.tril(torch.ones(block_len, block_len, device=q.device, dtype=torch.bool))
                qk = qk * causal_mask.float()
                o_intra = torch.matmul(qk, v_block)  # [block_len, e]

                # Inter-block: use accumulated kv from previous blocks
                o_inter = torch.matmul(q_block, kv)  # [block_len, e]

                o[bi, hi, start:end] = (o_intra + o_inter).to(q.dtype)

                # Update kv state
                kv = kv + torch.matmul(k_block.T, v_block)  # [d, e]

        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v = ctx.saved_tensors
        b, h, n, d = q.shape
        e = v.shape[-1]
        BLOCK = 64
        NUM_BLOCK = (n + BLOCK - 1) // BLOCK

        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)

        for bh_idx in range(b * h):
            bi = bh_idx // h
            hi = bh_idx % h

            # Intra-block gradients
            for blk in range(NUM_BLOCK):
                start = blk * BLOCK
                end = min(start + BLOCK, n)
                block_len = end - start

                q_block = q[bi, hi, start:end].float()
                k_block = k[bi, hi, start:end].float()
                v_block = v[bi, hi, start:end].float()
                do_block = do[bi, hi, start:end].float()

                # Causal mask
                causal_mask = torch.tril(torch.ones(block_len, block_len, device=q.device, dtype=torch.float32))

                # dqk = do @ v^T, masked
                dqk = torch.matmul(do_block, v_block.T) * causal_mask
                dq_intra = torch.matmul(dqk, k_block)

                # dk_intra
                dk_intra = torch.matmul(dqk.T, q_block)

                # dv_intra: qk^T (lower triangular transposed = upper triangular) @ do
                qk_trans = torch.matmul(k_block, q_block.T)
                qk_trans = qk_trans * torch.tril(torch.ones(block_len, block_len, device=q.device, dtype=torch.float32)).T
                dv_intra = torch.matmul(qk_trans, do_block)

                dq[bi, hi, start:end] = dq_intra.to(q.dtype)
                dk[bi, hi, start:end] = dk_intra.to(k.dtype)
                dv[bi, hi, start:end] = dv_intra.to(v.dtype)

            # Inter-block dq: forward accumulation of kv
            kv_trans = torch.zeros(e, d, dtype=torch.float32, device=q.device)
            for i in range(NUM_BLOCK):
                start = i * BLOCK
                end = min(start + BLOCK, n)

                if i > 0:
                    do_block = do[bi, hi, start:end].float()
                    dq_inter = torch.matmul(do_block, kv_trans)
                    dq[bi, hi, start:end] += dq_inter.to(q.dtype)

                v_block = v[bi, hi, start:end].float()
                k_block = k[bi, hi, start:end].float()
                kv_trans = kv_trans + torch.matmul(v_block.T, k_block)

            # Inter-block dk, dv: backward accumulation of dkv
            dkv = torch.zeros(d, e, dtype=torch.float32, device=q.device)
            for i in range(NUM_BLOCK - 1, -1, -1):
                start = i * BLOCK
                end = min(start + BLOCK, n)

                if i < NUM_BLOCK - 1:
                    k_block = k[bi, hi, start:end].float()
                    v_block = v[bi, hi, start:end].float()
                    dk_inter = torch.matmul(dkv, v_block.T).T  # [block_len, d]
                    dv_inter = torch.matmul(k_block, dkv)  # [block_len, e]
                    dk[bi, hi, start:end] += dk_inter.to(k.dtype)
                    dv[bi, hi, start:end] += dv_inter.to(v.dtype)

                do_block = do[bi, hi, start:end].float()
                q_block = q[bi, hi, start:end].float()
                dkv = dkv + torch.matmul(q_block.T, do_block)

        return dq, dk, dv


torch_lightning_attention = TorchLightningAttention2NoDecay.apply
