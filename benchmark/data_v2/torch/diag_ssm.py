import torch

class TorchDiagSSM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, s, x, Lambda):
        """
        s: [batch_size, dim] or [batch_size, dim] complex
        x: [length, batch_size, dim] or [length, batch_size, dim] complex
        Lambda: [dim] or [dim] complex
        """
        length, batch_size, dim = x.shape
        y = torch.zeros_like(x)

        if torch.is_complex(Lambda):
            s_t = s.clone()
            for t in range(length):
                y[t] = s_t * Lambda + x[t]
                s_t = y[t]
        else:
            s_t = s.clone()
            for t in range(length):
                y[t] = s_t * Lambda + x[t]
                s_t = y[t]

        ctx.save_for_backward(s, y, Lambda)
        return y

    @staticmethod
    def backward(ctx, grad_y):
        s, y, Lambda = ctx.saved_tensors
        length, batch_size, dim = y.shape

        grad_s = torch.zeros_like(s)
        grad_x = torch.zeros_like(grad_y)
        grad_Lambda = torch.zeros_like(Lambda, dtype=grad_y.dtype)

        if torch.is_complex(Lambda):
            # grad_Lambda_accum has shape [batch_size, dim] to accumulate per-batch
            grad_Lambda_accum = torch.zeros_like(s)
            grad_s_t = torch.zeros_like(s)
            for t in reversed(range(length)):
                grad_s_t = grad_y[t] + grad_s_t
                grad_x[t] = grad_s_t
                grad_Lambda_accum += (grad_s_t * s if t == 0 else grad_s_t * y[t-1])
                grad_s_t = grad_s_t * Lambda
            grad_s = grad_s_t
            grad_Lambda = grad_Lambda_accum.sum(dim=0)
        else:
            grad_Lambda_accum = torch.zeros_like(s)
            grad_s_t = torch.zeros_like(s)
            for t in reversed(range(length)):
                grad_s_t = grad_y[t] + grad_s_t
                grad_x[t] = grad_s_t
                grad_Lambda_accum += (grad_s_t * s if t == 0 else grad_s_t * y[t-1])
                grad_s_t = grad_s_t * Lambda
            grad_s = grad_s_t
            grad_Lambda = grad_Lambda_accum.sum(dim=0)

        return grad_s, grad_x, grad_Lambda

torch_diag_ssm = TorchDiagSSM.apply
