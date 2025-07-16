
### Note all the results are accurate but this is still in research phase so one can expects even better results than this ------.
# VIT_SKETCHING
This is Major project under Dr. Rachit Chhaya 
# Vision Transformer with Sketching Techniques

An efficient implementation of Vision Transformer (ViT) using attention sketching for reduced computational complexity.

## Key Features

### Core Components
- **Patch Embedding**: Convert images to patch sequences
- **Positional Encoding**: Learnable position embeddings
- **Class Token**: For classification task
- **Transformer Blocks**: With layer normalization and MLP

### Attention Mechanisms
- **Full Attention**: Standard self-attention
- **Sketched Attention**: Memory-efficient variants:
  - **Gaussian Sketching**: Random projection matrices
  - **Count Sketching**: Optimized hashing-based method

### Two Operational Modes
1. **Training Mode**:
   - Uses full attention matrices
   - Applies sketching during forward pass
   - Supports gradient backpropagation

2. **Inference Mode**:
   - Uses pre-computed sketched weights
   - Significant speedup at test time
   - accuracy increases with 50% sketching
   - have also use learnable sketch which results in even higher accuracy when the Q and K is sketched to 33.33% 

## Implementation Details

### SketchSelfAttention Class
```python
class SketchSelfAttention(nn.Module):
    def __init__(self, dim=64, sketch_dim=32, use_sketch=False,
                 train_mode=False, wq_path=None, wk_path=None,
                 s_q_path=None, s_k_path=None):
        # Supports both training and inference modes
        # Can load pre-computed sketch matrices
