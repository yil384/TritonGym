---
name: Operator request
about: Suggest a new GPU kernel to add to the benchmark
labels: operator
---

**Operator name**
e.g. `flash_attention_v2`, `int4_dequant_matmul`

**Category**
- [ ] Standard (common GPU primitive)
- [ ] OOD (less likely to be in pretraining corpora)
- [ ] DSL (Gluon / TLX)

**PyTorch reference (if you have one)**
```python
def torch_<op_name>(...):
    ...
```

**Why this op?**
What real workload uses it? Why is it interesting for an LLM to generate?

**Input shapes you'd want evaluated**
e.g. `[[(1, 32, 4096, 128), (1, 32, 4096, 128), (1, 32, 4096, 128)], ...]`
