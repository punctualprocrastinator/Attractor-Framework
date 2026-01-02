"""
Spectral-Projection Extensions for Mechanistic Interpretability
================================================================

This module extends the spectral_mi.py library with projection theory methods.
Combines spectral decomposition (SVD) with harmonic analysis (FFT) to analyze
LLM activations through the lens of orthogonal projections.

Mathematical Foundation:
    - SVD decomposes operators into orthogonal projection operators
    - Each rank-1 component P_i = σ_i(u_i ⊗ v_i^T) is a projection
    - Spectral Theorem: A = Σ_i σ_i P_i (resolution of identity)
    - Harmonic analysis measures periodicity in projected signals

Usage:
    >>> from spectral_projection_ext import multi_scale_projection_analysis
    >>> results = multi_scale_projection_analysis(model, prompt, layer=16, head=14)
    >>> print(results)
"""

import torch
import numpy as np
from spectral_mi import (
    calc_spectral_entropy_torch,
    calc_relative_entropy,
    decompose_attention_head_svd
)


# ==============================================================================
# MULTI-SCALE PROJECTION ANALYSIS
# ==============================================================================

def multi_scale_projection_analysis(model, prompt, layer_idx, head_idx, 
                                    hf_model=None, max_ranks=10, max_tokens=15):
    """
    Analyze spectral rigidity at multiple projection scales.
    Shows how rigidity emerges from hierarchical subspace structure.
    
    Mathematical Principle:
        For SVD: A = Σ_i σ_i(u_i ⊗ v_i^T)
        Each cumulative projection P_k = Σ_{i=1}^k σ_i(u_i ⊗ v_i^T)
        defines a k-dimensional subspace. We measure rigidity in each.
    
    Args:
        model: nnsight model
        prompt: Input text
        layer_idx: Layer number
        head_idx: Head number
        hf_model: HuggingFace model for weight access
        max_ranks: Maximum number of singular components to analyze
        max_tokens: Generation length
        
    Returns:
        List of dicts with rigidity at each scale
    """
    # Get SVD decomposition
    svd = decompose_attention_head_svd(model, layer_idx, head_idx, hf_model)
    U, S, V = svd['OV']['U'], svd['OV']['S'], svd['OV']['V']
    
    # Generate and trace
    print(f"Generating from: '{prompt}'...")
    with model.generate(prompt, max_new_tokens=max_tokens):
        out = model.generator.output.save()
    full_text = model.tokenizer.decode(out[0])
    
    with model.trace(full_text):
        acts = model.model.layers[layer_idx].input[0].save()
    val = acts.value if hasattr(acts, 'value') else acts
    acts = val[0] if val.ndim == 3 else val
    acts = acts.detach().cpu().float()
    
    results = []
    
    # Analyze cumulative projections
    for k in range(1, min(max_ranks + 1, len(S))):
        # Project onto top-k subspace
        P_k = torch.zeros(U.shape[0], U.shape[0])
        for i in range(k):
            P_k += S[i] * torch.outer(U[:, i], V[:, i])
        
        projected_k = acts @ P_k.T
        
        # Also analyze the k-th component alone
        P_k_only = S[k-1] * torch.outer(U[:, k-1], V[:, k-1])
        projected_k_only = acts @ P_k_only.T
        
        results.append({
            'rank': k,
            'cumulative_rigidity': calc_relative_entropy(projected_k),
            'component_rigidity': calc_relative_entropy(projected_k_only),
            'cumulative_variance': (S[:k].sum() / S.sum()).item(),
            'singular_value': S[k-1].item()
        })
    
    return results


# ==============================================================================
# ORTHOGONAL DECOMPOSITION
# ==============================================================================

def decompose_rigidity_orthogonally(acts, U, S, V, num_components=5):
    """
    Decompose signal into orthogonal subspaces and measure contribution
    to total rigidity.
    
    Mathematical Principle:
        Hilbert space decomposition: H = E_1 ⊕ E_2 ⊕ ... ⊕ E_n ⊕ E_⊥
        where E_i = span{u_i} are eigenspaces (1D in this case)
        and E_⊥ is the orthogonal complement.
    
    Args:
        acts: Activation tensor [seq_len, hidden_dim]
        U, S, V: SVD components from attention head
        num_components: Number of singular directions to decompose
        
    Returns:
        Dict with rigidity decomposition
    """
    # Total rigidity
    total_rigidity = calc_relative_entropy(acts)
    
    # Component rigidities
    component_rigidities = []
    cumulative_projection = torch.zeros_like(acts)
    
    for i in range(min(num_components, len(S))):
        # i-th projection operator: P_i = σ_i(u_i ⊗ v_i^T)
        P_i = S[i] * torch.outer(U[:, i], V[:, i])
        component_i = acts @ P_i.T
        cumulative_projection += component_i
        
        # Rigidity of this component
        r_i = calc_relative_entropy(component_i)
        
        # Rigidity of residual (orthogonal complement)
        residual = acts - cumulative_projection
        r_residual = calc_relative_entropy(residual)
        
        component_rigidities.append({
            'component': i,
            'singular_value': S[i].item(),
            'component_rigidity': r_i,
            'residual_rigidity': r_residual,
            'variance_explained': (S[:i+1].sum() / S.sum()).item()
        })
    
    return {
        'total_rigidity': total_rigidity,
        'components': component_rigidities,
        'decomposition_quality': sum(c['component_rigidity'] * c['singular_value'] 
                                    for c in component_rigidities) / (total_rigidity + 1e-8)
    }


# ==============================================================================
# PROJECTION-BASED STEERING
# ==============================================================================

def spectral_projection_steering(model, layer_idx, head_idx, 
                                 target_prompts, baseline_prompts,
                                 steering_ranks=[0, 1, 2], hf_model=None):
    """
    Create steering vectors in the principal subspace of a head.
    More robust than arbitrary direction steering.
    
    Mathematical Principle:
        Instead of steering in arbitrary directions, we constrain
        steering to the principal subspace P = span{u_0, u_1, ..., u_k}
        defined by top singular vectors. This is more stable and
        interpretable.
    
    Args:
        model: nnsight model
        layer_idx: Layer to steer
        head_idx: Head whose subspace to use
        target_prompts: List of prompts representing target behavior
        baseline_prompts: List of prompts representing baseline behavior
        steering_ranks: Which singular components to include in subspace
        hf_model: HuggingFace model for weight access
        
    Returns:
        Steering vector (torch.Tensor) on model.device
    """
    # Get head's principal subspace
    svd = decompose_attention_head_svd(model, layer_idx, head_idx, hf_model)
    U, S, V = svd['OV']['U'], svd['OV']['S'], svd['OV']['V']
    
    # Create projection operator for steering subspace
    P_steer = torch.zeros(U.shape[0], U.shape[0])
    for r in steering_ranks:
        P_steer += S[r] * torch.outer(U[:, r], V[:, r])
    
    # Collect target and baseline activations
    target_acts = []
    for p in target_prompts:
        try:
            with model.trace(p):
                act = model.model.layers[layer_idx].output[0].save()
            val = act.value if hasattr(act, 'value') else act
            target_acts.append(val[0, -1, :].cpu())
        except:
            pass
    
    baseline_acts = []
    for p in baseline_prompts:
        try:
            with model.trace(p):
                act = model.model.layers[layer_idx].output[0].save()
            val = act.value if hasattr(act, 'value') else act
            baseline_acts.append(val[0, -1, :].cpu())
        except:
            pass
    
    if not target_acts or not baseline_acts:
        return None
    
    # Compute difference vector
    target_mean = torch.stack(target_acts).mean(0)
    baseline_mean = torch.stack(baseline_acts).mean(0)
    diff_vector = target_mean - baseline_mean
    
    # Project steering vector onto principal subspace
    steering_vector = diff_vector @ P_steer.T
    steering_vector = steering_vector / (steering_vector.norm() + 1e-8)
    
    return steering_vector.to(model.device)


# ==============================================================================
# SIGNAL DECOMPOSITION ANALYSIS
# ==============================================================================

def analyze_signal_decomposition(acts, P_direction):
    """
    Decompose into parallel and orthogonal components.
    
    Mathematical Principle:
        For any projection P, we have the orthogonal decomposition:
        x = Px + (I-P)x = x_parallel + x_perpendicular
        
        These components are orthogonal: <x_parallel, x_perpendicular> = 0
    
    Args:
        acts: Activation tensor
        P_direction: Projection matrix
        
    Returns:
        Dict with rigidity of parallel and orthogonal components
    """
    signal_parallel = acts @ P_direction
    signal_orthogonal = acts - signal_parallel
    
    rigidity_parallel = calc_relative_entropy(signal_parallel)
    rigidity_orthogonal = calc_relative_entropy(signal_orthogonal)
    
    return {
        'parallel_rigidity': rigidity_parallel,
        'orthogonal_rigidity': rigidity_orthogonal,
        'rigidity_ratio': rigidity_parallel / (rigidity_orthogonal + 1e-8),
        'parallel_component': signal_parallel,
        'orthogonal_component': signal_orthogonal
    }


# ==============================================================================
# PROJECTION-VALUED SPECTRAL ENTROPY
# ==============================================================================

def projection_valued_entropy_measure(model, prompt, layer_idx, 
                                     num_projections=10, hf_model=None,
                                     max_tokens=15):
    """
    Measure how spectral entropy varies across different projection operators.
    
    Mathematical Principle:
        This creates a "projection-valued measure" μ that assigns an entropy
        value to each projection operator P_i. Related to spectral measures
        in functional analysis.
        
        μ(P_i) = Entropy(P_i · activations)
    
    Args:
        model: nnsight model
        prompt: Input text
        layer_idx: Layer to analyze
        num_projections: Number of singular directions to measure
        hf_model: HuggingFace model
        max_tokens: Generation length
        
    Returns:
        Dict mapping head_idx -> list of entropy measures
    """
    # Generate
    with model.generate(prompt, max_new_tokens=max_tokens):
        out = model.generator.output.save()
    full_text = model.tokenizer.decode(out[0])
    
    # Trace
    with model.trace(full_text):
        acts = model.model.layers[layer_idx].output[0].save()
    val = acts.value if hasattr(acts, 'value') else acts
    acts = val[0].detach().cpu().float()
    
    num_heads = hf_model.config.num_attention_heads
    head_dim = acts.shape[-1] // num_heads
    
    entropy_map = {}
    
    for head_idx in range(num_heads):
        # Get SVD for this head
        svd = decompose_attention_head_svd(model, layer_idx, head_idx, hf_model)
        U, S, V = svd['OV']['U'], svd['OV']['S'], svd['OV']['V']
        
        # Create projection-valued entropy measure
        pv_entropy = []
        for k in range(min(num_projections, len(S))):
            # Project onto k-th singular direction
            P_k = S[k] * torch.outer(U[:, k], V[:, k])
            projected = acts @ P_k.T
            entropy_k = calc_relative_entropy(projected)
            
            pv_entropy.append({
                'projection_rank': k,
                'singular_value': S[k].item(),
                'entropy': entropy_k,
                'weighted_entropy': entropy_k * S[k].item()  # Weight by importance
            })
        
        entropy_map[f'head_{head_idx}'] = pv_entropy
    
    return entropy_map


# ==============================================================================
# EFFECTIVE RANK ANALYSIS
# ==============================================================================

def compute_effective_rank(S, threshold=0.99):
    """
    Compute effective rank of a matrix from its singular values.
    
    Mathematical Principle:
        Effective rank measures the "true" dimensionality of a matrix,
        accounting for small singular values.
        
        Method 1: r_eff = (Σ σ_i)^2 / Σ σ_i^2
        Method 2: r_eff = min{k : Σ_{i≤k} σ_i ≥ threshold * Σ σ_i}
    
    Args:
        S: Singular values tensor
        threshold: Variance threshold for Method 2
        
    Returns:
        Dict with different effective rank measures
    """
    S_np = S.detach().cpu().numpy() if isinstance(S, torch.Tensor) else S
    
    # Method 1: Ratio of squared sums
    r_eff_1 = (S_np.sum() ** 2) / (S_np ** 2).sum()
    
    # Method 2: Variance threshold
    cumsum = np.cumsum(S_np) / S_np.sum()
    r_eff_2 = np.searchsorted(cumsum, threshold) + 1
    
    # Method 3: Entropy-based (information theoretic)
    p = S_np / S_np.sum()
    entropy = -np.sum(p * np.log(p + 1e-12))
    r_eff_3 = np.exp(entropy)
    
    return {
        'effective_rank_ratio': r_eff_1,
        'effective_rank_threshold': r_eff_2,
        'effective_rank_entropy': r_eff_3,
        'total_singular_values': len(S_np),
        'singular_value_distribution': S_np[:20].tolist()
    }


# ==============================================================================
# COMPLETE ANALYSIS PIPELINE
# ==============================================================================

def complete_spectral_projection_analysis(model, hf_model, prompt, 
                                          layer_idx, head_idx,
                                          max_new_tokens=20,
                                          verbose=True):
    """
    Complete analysis combining spectral entropy and projection theory.
    
    This function runs a comprehensive suite of analyses:
    1. SVD decomposition
    2. Multi-scale projection rigidity
    3. Orthogonal rigidity decomposition
    4. Effective rank computation
    5. Signal decomposition
    
    Args:
        model: nnsight model
        hf_model: HuggingFace model
        prompt: Input text
        layer_idx: Layer number
        head_idx: Head number
        max_new_tokens: Generation length
        verbose: Print results
        
    Returns:
        Dict with complete analysis results
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"SPECTRAL-PROJECTION ANALYSIS")
        print(f"{'='*60}")
        print(f"Layer {layer_idx}, Head {head_idx}")
        print(f"Prompt: {prompt}\n")
    
    # 1. Generate text
    with model.generate(prompt, max_new_tokens=max_new_tokens):
        out = model.generator.output.save()
    full_text = model.tokenizer.decode(out[0])
    
    if verbose:
        print(f"Generated: {full_text}\n")
    
    # 2. SVD Decomposition
    svd = decompose_attention_head_svd(model, layer_idx, head_idx, hf_model)
    U, S, V = svd['OV']['U'], svd['OV']['S'], svd['OV']['V']
    
    # 3. Trace activations
    with model.trace(full_text):
        acts = model.model.layers[layer_idx].input[0].save()
    val = acts.value if hasattr(acts, 'value') else acts
    acts = val[0] if val.ndim == 3 else val
    acts = acts.detach().cpu().float()
    
    results = {
        'text': full_text,
        'prompt': prompt,
        'layer': layer_idx,
        'head': head_idx,
        'total_rigidity': calc_relative_entropy(acts),
        'singular_values': S[:10].tolist()
    }
    
    # 4. Multi-scale analysis
    if verbose:
        print("Multi-scale projection rigidity:")
    multiscale = multi_scale_projection_analysis(
        model, prompt, layer_idx, head_idx, hf_model, max_ranks=5
    )
    results['multiscale'] = multiscale
    
    if verbose:
        for m in multiscale:
            print(f"  Rank {m['rank']}: "
                  f"Cum. R={m['cumulative_rigidity']:.3f}, "
                  f"Comp. R={m['component_rigidity']:.3f}, "
                  f"Var={m['cumulative_variance']:.3f}")
    
    # 5. Orthogonal decomposition
    if verbose:
        print("\nOrthogonal rigidity decomposition:")
    decomp = decompose_rigidity_orthogonally(acts, U, S, V, num_components=5)
    results['decomposition'] = decomp
    
    if verbose:
        for c in decomp['components']:
            print(f"  Component {c['component']}: "
                  f"σ={c['singular_value']:.3f}, "
                  f"R={c['component_rigidity']:.3f}, "
                  f"Residual R={c['residual_rigidity']:.3f}")
    
    # 6. Effective rank
    rank_stats = compute_effective_rank(S)
    results['effective_rank'] = rank_stats
    
    if verbose:
        print(f"\nEffective rank (ratio method): {rank_stats['effective_rank_ratio']:.2f}")
        print(f"Effective rank (99% variance): {rank_stats['effective_rank_threshold']}")
        print(f"Effective rank (entropy method): {rank_stats['effective_rank_entropy']:.2f}")
        print(f"Total singular values: {rank_stats['total_singular_values']}")
        
        # Hypothesis test
        print(f"\n{'='*60}")
        print("HYPOTHESIS: High rigidity ↔ Low effective rank")
        print(f"Rigidity: {results['total_rigidity']:.3f}")
        print(f"Effective Rank: {rank_stats['effective_rank_ratio']:.2f}")
        print(f"Inverse correlation: {1.0 / (rank_stats['effective_rank_ratio'] + 1):.3f}")
        print(f"{'='*60}\n")
    
    return results


# ==============================================================================
# SCANNING AND DISCOVERY
# ==============================================================================

def scan_layer_for_rigid_projections(model, hf_model, prompt, layer_idx,
                                     max_tokens=15, top_k=5):
    """
    Scan all heads in a layer and rank by projection-based rigidity.
    Discovers which heads have the most structured projection patterns.
    
    Args:
        model: nnsight model
        hf_model: HuggingFace model
        prompt: Input text
        layer_idx: Layer to scan
        max_tokens: Generation length
        top_k: Number of top heads to return
        
    Returns:
        List of (head_idx, rigidity_score, details) sorted by rigidity
    """
    print(f"Scanning layer {layer_idx} for rigid projection patterns...")
    
    num_heads = hf_model.config.num_attention_heads
    
    head_results = []
    
    for head_idx in range(num_heads):
        try:
            # Analyze this head
            results = complete_spectral_projection_analysis(
                model, hf_model, prompt, layer_idx, head_idx,
                max_new_tokens=max_tokens, verbose=False
            )
            
            # Compute composite score
            score = (results['total_rigidity'] * 
                    (1.0 / (results['effective_rank']['effective_rank_ratio'] + 1)))
            
            head_results.append({
                'head': head_idx,
                'score': score,
                'rigidity': results['total_rigidity'],
                'effective_rank': results['effective_rank']['effective_rank_ratio'],
                'top_singular_value': results['singular_values'][0]
            })
            
        except Exception as e:
            print(f"  Error on head {head_idx}: {e}")
            continue
    
    # Sort by score
    head_results.sort(key=lambda x: x['score'], reverse=True)
    
    print(f"\nTop {top_k} heads by rigidity-rank score:")
    for i, h in enumerate(head_results[:top_k]):
        print(f"  {i+1}. Head {h['head']}: "
              f"Score={h['score']:.3f}, "
              f"R={h['rigidity']:.3f}, "
              f"Eff.Rank={h['effective_rank']:.2f}")
    
    return head_results


# ==============================================================================
# EXAMPLE USAGE
# ==============================================================================
if __name__ == "__main__":
    print("Spectral-Projection Extensions loaded.")
    print("\nExample usage:")
    print("  results = complete_spectral_projection_analysis(model, hf_model, prompt, 16, 14)")
    print("  scan_results = scan_layer_for_rigid_projections(model, hf_model, prompt, 16)")
    print("  steering_vec = spectral_projection_steering(model, 16, 14, targets, baselines)")
