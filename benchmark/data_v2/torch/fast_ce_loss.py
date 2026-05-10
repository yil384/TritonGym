import torch
import torch.nn.functional as F

def torch_fast_ce_loss(logits, logit_softcapping=0.0, logit_scaling=0.0, n_items=None):
    """
    Pure PyTorch implementation of fast cross-entropy (from Triton version)
    Supports optional logit softcapping and scaling.
    logits: [batch, seq_len, vocab_size]
    labels: [batch, seq_len]
    """
    batch, seq_len, vocab_size = logits.shape
    labels = torch.randint(0, vocab_size, (batch, seq_len), device=logits.device, dtype=torch.long)
    logits_flat = logits.view(batch*seq_len, vocab_size)
    labels_flat = labels.view(-1)

    # Apply logit scaling first, then softcapping (matches Triton order)
    if logit_scaling != 0:
        logits_flat = logits_flat * logit_scaling

    if logit_softcapping != 0:
        logits_flat = logit_softcapping * torch.tanh(logits_flat / logit_softcapping)

    # Compute cross-entropy, ignore index -100
    loss = F.cross_entropy(logits_flat, labels_flat, reduction='none', ignore_index=-100)

    if n_items is None:
        n_items = (labels_flat != -100).sum()
    return loss.sum() / n_items
