"""
Spectral Mechanistic Interpretability (Spectral MI) Library
===========================================================

A toolkit for analyzing Large Language Models through the lens of Spectral Dynamics.
This library provides tools to measure "Spectral Rigidity" (entropy of hidden states),
decompose attention heads via SVD, and perform spectral steering interventions.

Dependencies:
    - torch
    - numpy
    - nnsight (for model tracing)
    - transformers (for weight access)

Usage with nnsight:
    >>> from spectral_mi import calc_relative_entropy
    >>> with model.trace("Prompt") as tracer:
    >>>     acts = model.model.layers[16].output[0].save()
    >>> print(calc_relative_entropy(acts))
"""

import torch
import numpy as np
import torch.nn.functional as F
from nnsight import LanguageModel
import warnings

# ==============================================================================
# 1. CORE METRICS
# ==============================================================================

def calc_spectral_entropy_torch(activations):
    """
    Computes the Shannon Entropy of the power spectrum of a sequence of activations.
    
    Args:
        activations (torch.Tensor): Shape [seq_len, hidden_dim] or [batch, seq_len, hidden_dim].
        
    Returns:
        float: Average spectral entropy across channel dimensions.
    """
    if not isinstance(activations, torch.Tensor):
        if hasattr(activations, 'value'): # Handle nnsight Proxy
            activations = activations.value
        else:
            activations = torch.tensor(activations)
            
    activations = activations.float().detach()
    
    # Handle dimensions: Ensure [batch, dim, seq_len] for FFT logic efficiency or standardize
    # Standard approach used in experiments: Mean over hidden dim, then FFT over sequence.
    # Wait, the previous efficient implementation was:
    # 1. Mean over hidden dimension -> [seq_len, 1] (Average trajectory)
    # 2. RFFT over sequence dim.
    
    if activations.ndim == 2:
        activations = activations.unsqueeze(0) # [1, seq, dim]
        
    # Method: Average Hidden Projection (Robust for Rigidity)
    # Project to 1D signal per sample
    activations_avg = activations.mean(dim=-1, keepdim=True) # [batch, seq, 1]
    
    # FFT over sequence dimension (dim=1)
    fft_vals = torch.fft.rfft(activations_avg.squeeze(-1), dim=1)
    
    # Power Spectrum
    power = fft_vals.real**2 + fft_vals.imag**2
    power_sum = torch.sum(power, dim=1, keepdim=True) + 1e-12
    
    # Probability Distribution
    p = power / power_sum
    
    # Shannon Entropy
    entropy = -torch.sum(p * torch.log(p + 1e-12), dim=1)
    
    return entropy.mean()

def calc_relative_entropy(activations, window_size=5):
    """
    Computes 'Spectral Rigidity' (R), a normalized metric (0 to 1).
    R = 1.0 means the signal is fully periodic/rigid (low entropy).
    R = 0.0 means the signal is random/unstructured (high entropy).
    
    Args:
        activations: Input tensor.
        window_size (int): Size of the sliding window (acts as effective sequence length normalization).
    
    Returns:
        float: Rigidity score (0.0 to 1.0).
    """
    # Materialize nnsight proxy
    if hasattr(activations, 'value'):
        activations = activations.value
    
    # Check valid tensor
    if not isinstance(activations, torch.Tensor) or activations.numel() == 0:
        return 0.0
        
    # Dimensions
    if activations.ndim == 2:
        activations = activations.unsqueeze(0)
        
    # Windowing (Focus on recent dynamics)
    act_len = activations.shape[1]
    
    # If sequence is shorter than window, usage full length
    effective_window = min(act_len, window_size)
    
    # Slice last 'window_size' tokens
    if act_len > window_size:
        activations = activations[:, -window_size:, :]
        
    # Calculate Raw Entropy (S)
    raw_entropy = calc_spectral_entropy_torch(activations).item()
    
    # Calculate Max Entropy (S_max) for this window size (Uniform distribution)
    # FFT length for rfft is (N/2) + 1
    n_freqs = (effective_window // 2) + 1
    max_entropy = np.log(n_freqs)
    
    # Relative Entropy (Rigidity): 1 - S/S_max
    if max_entropy > 0:
        relative_entropy = 1.0 - (raw_entropy / max_entropy)
    else:
        relative_entropy = 0.0
        
    return max(0.0, min(1.0, relative_entropy))

# ==============================================================================
# 2. SVD ANALYSIS TOOLS
# ==============================================================================

def decompose_attention_head_svd(model, layer_idx, head_idx, hf_model=None):
    """
    Decomposes an Attention Head (OV and QK circuits) into Singular Components.
    
    Args:
        model: nnsight.LanguageModel or HuggingFace model.
        layer_idx (int): Layer number.
        head_idx (int): Head number.
        hf_model: Optional explicit HF model (recommended for weight access).
        
    Returns:
        dict: {'OV': {'U', 'S', 'V'}, 'QK': ...}
    """
    # Prefer explicit HF model to avoid nnsight meta-tensor issues on CPU SVD
    source_model = hf_model if hf_model is not None else model
    
    # Handle different wrapper structures
    if hasattr(source_model, 'model'): 
         layers = source_model.model.layers
    else:
         layers = source_model.layers # Fallback
         
    # Extract Weights
    with torch.no_grad():
        # Cast to float32 for CPU SVD compatibility
        W_V = layers[layer_idx].self_attn.v_proj.weight.cpu().float()
        W_O = layers[layer_idx].self_attn.o_proj.weight.cpu().float()
    
    # Determine Head Dimensions
    config = source_model.config
    num_heads = config.num_attention_heads
    d_model = config.hidden_size
    d_head = d_model // num_heads
    
    # Slice Head Weights
    start_idx = head_idx * d_head
    end_idx = (head_idx + 1) * d_head
    
    W_V_head = W_V[start_idx:end_idx, :] # [d_head, d_model]
    W_O_head = W_O[:, start_idx:end_idx] # [d_model, d_head]
    
    # OV Circuit Matrix
    W_OV = W_O_head @ W_V_head # [d_model, d_model]
    
    # SVD
    # Full matrices=False gives U:[M,K], S:[K], Vh:[K,N]
    OV_U, OV_S, OV_Vh = torch.linalg.svd(W_OV, full_matrices=False)
    
    return {
        'OV': {'U': OV_U, 'S': OV_S, 'V': OV_Vh.T}, # V is Transpose of Vh
        'QK': None # Not implemented for spectral analysis yet
    }

def analyze_direction_entropy(model, prompt, layer_idx, head_idx, direction_idx,
                              component='OV', max_tokens=15, hf_model=None):
    """
    Analyzes the spectral rigidity of a specific Singular Direction in an Attention Head.
    Generates text and projects activations onto the direction vector.
    
    Args:
        model: nnsight model.
        prompt: input text.
        direction_idx: Rank of the singular vector (0-9 typically).
    """
    # 1. Decompose
    svd = decompose_attention_head_svd(model, layer_idx, head_idx, hf_model=hf_model)
    
    # 2. Reconstruct Direction Vector (W_dir = sigma * u * v^T)
    U, S, V = svd[component]['U'], svd[component]['S'], svd[component]['V']
    
    sigma = S[direction_idx].item()
    u = U[:, direction_idx] # Output direction
    v = V[:, direction_idx] # Input direction
    
    # Effective Transformation Matrix for this rank-1 component
    W_direction = sigma * torch.outer(u, v) # [d_model, d_model]
    
    # 3. Trace and Project
    # Note: Efficient Scan uses generate-then-trace
    
    print(f"Generating from propmt: '{prompt}'...")
    with model.generate(prompt, max_new_tokens=max_new_tokens) as generator:
        out = model.generator.output.save()
        
    full_text = model.tokenizer.decode(out[0])
    
    # Trace full sequence
    with model.trace(full_text) as tracer:
        # Save input to O_proj (This is the post-Attention, pre-Output activation?) 
        # Actually, standard analysis typically looks at the RESULT of the head.
        # But for OV analysis, we often project the INPUT to the head (x) via W_OV.
        # Here, let's look at the HEAD OUTPUT contribution.
        # Head Output = Attention(x) @ W_OV
        # The SVD decomposes W_OV.
        # So we want to project the input-to-OV-circuit onto the direction.
        
        # Accessing layer input is easiest.
        layer_in = model.model.layers[layer_idx].input[0].save()
        
    acts = layer_in.value if hasattr(layer_in, 'value') else layer_in
    if isinstance(acts, tuple): acts = acts[0]
    acts = acts.detach().cpu().float()
    if acts.ndim == 3: acts = acts[0]
    
    # Project: x @ W_dir.T
    projected = acts @ W_direction.T
    
    # Measure Rigidity of this component
    r = calc_relative_entropy(projected)
    
    return {
        'rigidity': r,
        'direction_idx': direction_idx,
        'sigma': sigma,
        'text': full_text
    }

def scan_layer_heads(model, model_hf, layer_idx, prompt, max_new_tokens=15):
    """
    Scans all heads in a layer to find the one with highest spectral rigidity.
    Useful for discovering "Enforcer Heads".
    """
    print(f"Scanning Layer {layer_idx} for high-rigidity heads...")
    
    # 1. Generate
    with model.generate(prompt, max_new_tokens=max_new_tokens):
        out = model.generator.output.save()
    full_text = model.tokenizer.decode(out[0])
    
    # 2. Trace Output Projection Input (Activations just before O_proj)
    # This captures the concatenation of all head outputs (before mixing)
    # shape: [batch, seq, num_heads * head_dim]
    with model.trace(full_text):
         attn_out = model.model.layers[layer_idx].self_attn.o_proj.input[0].save()
         
    acts = attn_out.value if hasattr(attn_out, 'value') else attn_out
    if acts.ndim == 3: acts = acts[0]
    acts = acts.detach().cpu().float()
    
    num_heads = model_hf.config.num_attention_heads
    head_dim = acts.shape[-1] // num_heads
    
    head_stats = []
    
    for h in range(num_heads):
        # Slice head
        start = h * head_dim
        end = (h+1) * head_dim
        chunk = acts[:, start:end]
        
        r = calc_relative_entropy(chunk)
        head_stats.append(r)
        
    best_head = np.argmax(head_stats)
    return best_head, head_stats[best_head], head_stats

# ==============================================================================
# 3. STEERING TOOLS
# ==============================================================================

def get_steering_vector_multi_source(model, layer, refusal_prompts, harmless_prompts):
    """
    Extracts a robust steering vector by averaging (Refusal - Harmless) across multiple examples.
    """
    targets = []
    baselines = []
    
    print(f"   [Debug] Layer {layer} extraction started.")
    
    # Collect Targets
    for p in refusal_prompts:
        try:
            with model.trace(p, validate=False, scan=False):
                act = model.model.layers[layer].output[0].save()
            
            # Access value safely
            val = act.value if hasattr(act, 'value') else act
            if isinstance(val, tuple): val = val[0]
            
            # [batch, seq, hidden]
            if val.ndim == 3: v = val[0, -1, :]
            elif val.ndim == 2: v = val[-1] if val.shape[0] > 1 else val[0]
            else: 
                print(f"   [Debug] Unexpected shape {val.shape}")
                continue
                
            targets.append(v.detach().cpu())
        except Exception as e: 
            print(f"   [Debug] Error on refusal prompt '{p[:10]}...': {e}")
        
    # Collect Baselines
    for p in harmless_prompts:
        try:
            with model.trace(p, validate=False, scan=False):
                act = model.model.layers[layer].output[0].save()
                
            val = act.value if hasattr(act, 'value') else act
            if isinstance(val, tuple): val = val[0]
            
            if val.ndim == 3: v = val[0, -1, :]
            elif val.ndim == 2: v = val[-1] if val.shape[0] > 1 else val[0]
            else: 
                print(f"   [Debug] Unexpected shape {val.shape}")
                continue
                
            baselines.append(v.detach().cpu())
        except Exception as e: 
            print(f"   [Debug] Error on harmless prompt '{p[:10]}...': {e}")
        
    print(f"   [Debug] Extracted {len(targets)} targets and {len(baselines)} baselines.")
        
    if not targets or not baselines: return None
    
    vec = torch.stack(targets).mean(0) - torch.stack(baselines).mean(0)
    vec = vec / (vec.norm() + 1e-8)
    return vec.to(model.device)

# ==============================================================================
# EXAMPLE USAGE
# ==============================================================================
if __name__ == "__main__":
    print("Spectral MI Library loaded.")
    print("Example commands:")
    print("  entropy = calc_spectral_entropy_torch(my_tensor)")
    print("  rigidity = calc_relative_entropy(my_tensor)")
    print("  svd = decompose_attention_head_svd(model, 27, 14, hf_model=model_hf)")
