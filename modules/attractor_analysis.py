"""
Attractor Analysis Tools for MATS Project
==========================================

Practical implementations of projection theory methods for analyzing
safety and bias attractors in LLMs.

These tools extend your spectral_mi.py library with geometric methods
for extracting, measuring, and manipulating attractor subspaces.

Author: Based on Spectral MI + Projection Theory synthesis
"""

import torch
import numpy as np
from spectral_mi import (
    decompose_attention_head_svd,
    calc_relative_entropy
)


# ==============================================================================
# ATTRACTOR EXTRACTION
# ==============================================================================

def decompose_attention_head_svd_robust(model, layer_idx, head_idx, hf_model=None):
    """
    Robust decomposition function handling Grouped Query Attention (GQA).
    """
    source_model = hf_model if hf_model is not None else model
    if hasattr(source_model, 'model'): layers = source_model.model.layers
    else: layers = source_model.layers
         
    with torch.no_grad():
        W_V = layers[layer_idx].self_attn.v_proj.weight.cpu().float()
        W_O = layers[layer_idx].self_attn.o_proj.weight.cpu().float()
    
    config = source_model.config
    num_heads = config.num_attention_heads
    num_kv_heads = getattr(config, 'num_key_value_heads', num_heads)
    d_model = config.hidden_size
    d_head = d_model // num_heads
    group_size = num_heads // num_kv_heads

    # GQA Logic for V projection
    kv_head_idx = head_idx // group_size
    start_v = kv_head_idx * d_head
    end_v = (kv_head_idx + 1) * d_head
    
    W_V_head = W_V[start_v:end_v, :]
    
    # Standard Logic for O projection
    start_o = head_idx * d_head
    end_o = (head_idx + 1) * d_head
    W_O_head = W_O[:, start_o:end_o]
    
    W_OV = W_O_head @ W_V_head
    OV_U, OV_S, OV_Vh = torch.linalg.svd(W_OV, full_matrices=False)
    
    return {'OV': {'U': OV_U, 'S': OV_S, 'V': OV_Vh.T}}

def extract_attractor_subspace(model, hf_model, layer_idx, head_idx,
                               target_prompts, max_rank=50, variance_threshold=0.99):
    """
    Extract the geometric projection operator for a behavior attractor.
    
    This computes the subspace that activations exhibiting a target behavior
    project onto. The result is a projection matrix P that maps the full
    activation space R^d onto the lower-dimensional attractor subspace R^k.
    
    Mathematical Framework:
        For behavior B, we seek P such that:
        - x exhibiting B satisfies ||x - P(x)|| ≈ 0
        - P is idempotent: P² = P
        - P is self-adjoint: P* = P
        - dim(range(P)) << d_model
    
    Args:
        model: nnsight model
        hf_model: HuggingFace model for weight access
        layer_idx: Layer to analyze
        head_idx: Attention head to analyze
        target_prompts: List of prompts exhibiting target behavior
        max_rank: Maximum rank to consider
        variance_threshold: Fraction of variance to explain (e.g., 0.99)
        
    Returns:
        dict containing:
            - projection_matrix: P [d_model, d_model]
            - basis: Orthonormal basis for subspace [d_model, k]
            - effective_dim: k (dimensionality of subspace)
            - singular_values: Spectrum of the subspace
            - explained_variance: Fraction of variance captured
    """
    print(f"Extracting attractor from Layer {layer_idx}, Head {head_idx}")
    print(f"Using {len(target_prompts)} target prompts...")
    
    # Step 1: Get the head's transformation matrix (OV circuit)
    # Use robust local function instead of imported one to handle GQA
    svd = decompose_attention_head_svd_robust(model, layer_idx, head_idx, hf_model)
    U, S, V = svd['OV']['U'], svd['OV']['S'], svd['OV']['V']
    
    # Construct full OV matrix
    W_OV = U @ torch.diag(S) @ V.T  # [d_model, d_model]
    
    # Step 2: Collect activations for target prompts
    activations = []
    
    for i, prompt in enumerate(target_prompts):
        try:
            with model.trace(prompt):
                # Get activation at input to this layer
                act = model.model.layers[layer_idx].input[0].save()
            
            # Extract final token activation
            val = act.value if hasattr(act, 'value') else act
            x = val[0, -1, :] if val.ndim == 3 else val[-1, :]
            activations.append(x.cpu().float())
            
            if (i + 1) % 10 == 0:
                print(f"  Processed {i+1}/{len(target_prompts)} prompts")
                
        except Exception as e:
            print(f"  Warning: Failed on prompt {i}: {e}")
            continue
    
    if not activations:
        raise ValueError("Failed to collect any activations")
    
    # Step 3: Stack activations into matrix
    X = torch.stack(activations)  # [n_samples, d_model]
    print(f"Collected {X.shape[0]} activation samples")
    
    # Step 4: Transform through head
    X_transformed = X @ W_OV.T  # Apply head transformation
    
    # Step 5: Find principal subspace via SVD
    # This gives us the subspace that the head maps inputs to
    U_act, S_act, Vh_act = torch.linalg.svd(X_transformed, full_matrices=False)
    
    # Step 6: Determine effective dimensionality
    total_variance = (S_act ** 2).sum()
    cumsum = torch.cumsum(S_act ** 2, 0) / total_variance
    k = min(torch.searchsorted(cumsum, variance_threshold).item() + 1, max_rank)
    
    print(f"Effective dimensionality: {k} (explains {cumsum[k-1]*100:.1f}% variance)")
    
    # Step 7: Extract basis and projection matrix
    basis = Vh_act[:k, :].T  # [d_model, k] - orthonormal basis vectors
    P = basis @ basis.T      # [d_model, d_model] - projection operator
    
    # Step 8: Verify projection properties
    # Check idempotency: P² ≈ P
    P_squared = P @ P
    idempotency_error = torch.norm(P_squared - P) / torch.norm(P)
    
    # Check self-adjointness: P* ≈ P
    symmetry_error = torch.norm(P - P.T) / torch.norm(P)
    
    print(f"Projection quality:")
    print(f"  Idempotency error: {idempotency_error:.6f}")
    print(f"  Symmetry error: {symmetry_error:.6f}")
    
    return {
        'projection_matrix': P,
        'basis': basis,
        'effective_dim': k,
        'singular_values': S_act[:max_rank].detach().cpu().numpy(),
        'explained_variance': cumsum[k-1].item(),
        'head_transform': W_OV,
        'quality': {
            'idempotency_error': idempotency_error.item(),
            'symmetry_error': symmetry_error.item()
        }
    }


# ==============================================================================
# ATTRACTOR MEASUREMENT
# ==============================================================================

def measure_attractor_distance(activation, attractor_subspace):
    """
    Decompose activation into parallel and orthogonal components relative
    to an attractor subspace.
    
    Mathematical Framework:
        For projection P and activation x:
        x = x_∥ + x_⊥
        where x_∥ = P(x) (component in subspace)
              x_⊥ = (I-P)(x) (component orthogonal to subspace)
        
        These satisfy:
        - <x_∥, x_⊥> = 0 (orthogonality)
        - ||x||² = ||x_∥||² + ||x_⊥||² (Pythagoras)
    
    Args:
        activation: Activation vector [d_model]
        attractor_subspace: Dict from extract_attractor_subspace
        
    Returns:
        dict with decomposition metrics
    """
    P = attractor_subspace['projection_matrix']
    
    # Ensure same device and dtype
    if isinstance(activation, torch.Tensor):
        activation = activation.cpu().float()
    else:
        activation = torch.tensor(activation, dtype=torch.float32)
    
    # Decompose: x = x_∥ + x_⊥
    x_parallel = activation @ P        # Component in subspace
    x_orthogonal = activation - x_parallel  # Orthogonal component
    
    # Compute norms
    norm_total = torch.norm(activation)
    norm_parallel = torch.norm(x_parallel)
    norm_orthogonal = torch.norm(x_orthogonal)
    
    # Compute metrics
    alignment = norm_parallel / (norm_total + 1e-8)  # In [0, 1]
    distance = norm_orthogonal                        # Distance to subspace
    
    # Verify orthogonality (should be ~0)
    dot_product = torch.dot(x_parallel, x_orthogonal)
    orthogonality_check = abs(dot_product) / (norm_parallel * norm_orthogonal + 1e-8)
    
    return {
        'parallel_component': x_parallel,
        'orthogonal_component': x_orthogonal,
        'alignment': alignment.item(),          # How much is in subspace
        'distance': distance.item(),            # How far from subspace
        'norm_parallel': norm_parallel.item(),
        'norm_orthogonal': norm_orthogonal.item(),
        'norm_total': norm_total.item(),
        'orthogonality_check': orthogonality_check.item()  # Should be ~0
    }


def batch_measure_alignment(model, prompts, layer_idx, attractor_subspace):
    """
    Measure alignment with attractor for multiple prompts.
    
    Returns:
        List of alignment scores, one per prompt
    """
    alignments = []
    
    for prompt in prompts:
        try:
            with model.trace(prompt):
                act = model.model.layers[layer_idx].output[0].save()
            
            val = act.value if hasattr(act, 'value') else act
            x = val[0, -1, :].cpu() if val.ndim == 3 else val[-1, :].cpu()
            
            metrics = measure_attractor_distance(x, attractor_subspace)
            alignments.append(metrics['alignment'])
            
        except:
            alignments.append(np.nan)
    
    return alignments


# ==============================================================================
# TRAJECTORY ANALYSIS
# ==============================================================================

def trace_attractor_trajectory(model, prompt, attractor_subspace,
                               start_layer=0, end_layer=27, max_tokens=10):
    """
    Track how a representation approaches an attractor across layers.
    
    This reveals the dynamics of attractor formation - does the model
    gradually converge to the attractor, or does it snap to it at a
    specific layer?
    
    Args:
        model: nnsight model
        prompt: Input text
        attractor_subspace: Attractor to track
        start_layer: First layer to measure
        end_layer: Last layer to measure
        max_tokens: Number of tokens to generate
        
    Returns:
        dict with trajectory data
    """
    print(f"Tracing attractor trajectory for: '{prompt}'")
    
    # Generate text
    with model.generate(prompt, max_new_tokens=max_tokens):
        out = model.generator.output.save()
    
    out_val = out.value if hasattr(out, 'value') else out
    full_text = model.tokenizer.decode(out_val[0])
    
    print(f"Generated: {full_text}")
    
    # Trace all layers
    activations = {}
    with model.trace(full_text):
        for layer_idx in range(start_layer, end_layer + 1):
            act = model.model.layers[layer_idx].output[0].save()
            activations[layer_idx] = act
    
    # Compute metrics at each layer
    trajectory = {
        'layers': [],
        'alignments': [],
        'distances': [],
        'rigidity': []
    }
    
    for layer_idx in range(start_layer, end_layer + 1):
        act = activations[layer_idx]
        val = act.value if hasattr(act, 'value') else act
        x = val[0, -1, :].cpu() if val.ndim == 3 else val[-1, :].cpu()
        
        # Measure distance to attractor
        metrics = measure_attractor_distance(x, attractor_subspace)
        
        # Also measure spectral rigidity at this layer
        full_seq = val[0].cpu() if val.ndim == 3 else val.cpu()
        r = calc_relative_entropy(full_seq)
        
        trajectory['layers'].append(layer_idx)
        trajectory['alignments'].append(metrics['alignment'])
        trajectory['distances'].append(metrics['distance'])
        trajectory['rigidity'].append(r)
    
    trajectory['text'] = full_text
    trajectory['prompt'] = prompt
    
    return trajectory


# ==============================================================================
# PROJECTION-BASED STEERING
# ==============================================================================

def create_projection_steering_hook(attractor_subspace, steering_strength=1.0, 
                                    invert=False, target_position=-1):
    """
    Create a hook for projection-based steering.
    
    Mathematical Framework:
        For projection P:
        - Forward steering: x_new = α·P(x) + (1-α)·x
          (pulls toward attractor)
        - Inverse steering: x_new = x + β·(I-P)(x)
          (pushes away from attractor by amplifying orthogonal component)
    
    Args:
        attractor_subspace: Subspace to steer with/against
        steering_strength: Magnitude of steering (α or β)
        invert: If True, steer away from attractor; if False, toward it
        target_position: Which token position to steer (-1 = last)
        
    Returns:
        Hook function that can be registered
    """
    P = attractor_subspace['projection_matrix']
    
    def steering_hook(module, input, output):
        # Extract activations
        x = output[0] if isinstance(output, tuple) else output
        
        # Move P to same device and dtype as x
        P_device = P.to(device=x.device, dtype=x.dtype)
        
        # Work with the target position
        if x.ndim == 3:  # [batch, seq, hidden]
            x_target = x[:, target_position, :]
        else:  # [seq, hidden]
            x_target = x[target_position, :]
        
        if invert:
            # Anti-steering: Amplify component orthogonal to attractor
            # x_new = x + β·(I-P)(x) = x + β·x_⊥
            x_parallel = x_target @ P_device
            x_orthogonal = x_target - x_parallel
            x_steered = x_target + steering_strength * x_orthogonal
        else:
            # Pro-steering: Pull toward attractor
            # x_new = α·P(x) + (1-α)·x
            x_projected = x_target @ P_device
            x_steered = steering_strength * x_projected + (1 - steering_strength) * x_target
        
        # Replace in output
        if x.ndim == 3:
            x_new = x.clone()
            x_new[:, target_position, :] = x_steered
        else:
            x_new = x.clone()
            x_new[target_position, :] = x_steered
        
        if isinstance(output, tuple):
            return (x_new,) + output[1:]
        else:
            return x_new
    
    return steering_hook


def apply_projection_steering(model, layer_idx, attractor_subspace,
                              prompt, max_tokens=20,
                              steering_strength=1.0, invert=False):
    """
    Generate text with projection-based steering.
    
    Args:
        model: nnsight model
        layer_idx: Layer to apply steering at
        attractor_subspace: Subspace to use for steering
        prompt: Input prompt
        max_tokens: Number of tokens to generate
        steering_strength: How strong to steer
        invert: Steer away from (True) or toward (False) attractor
        
    Returns:
        Generated text
    """
    # Create and register hook
    hook_fn = create_projection_steering_hook(
        attractor_subspace, 
        steering_strength=steering_strength,
        invert=invert
    )
    
    handle = model.model.layers[layer_idx].register_forward_hook(hook_fn)
    
    try:
        # Generate with steering
        with model.generate(prompt, max_new_tokens=max_tokens):
            out = model.generator.output.save()
        
        out_val = out.value if hasattr(out, 'value') else out
        text = model.tokenizer.decode(out_val[0])
        
    finally:
        # Always remove hook
        handle.remove()
    
    return text


# ==============================================================================
# ATTRACTOR STRENGTH ANALYSIS
# ==============================================================================

def measure_attractor_strength(model, layer_idx, attractor_subspace,
                               target_prompts, perturbation_sizes=[0.1, 0.5, 1.0, 2.0]):
    """
    Measure how robust an attractor is to perturbations.
    
    Strong attractors should maintain high alignment even when perturbed.
    Weak attractors lose alignment quickly.
    
    Mathematical Framework:
        For attractor A with basin of attraction B_A, the strength is
        characterized by how far from A a point can be and still converge to A.
        
        We measure this by perturbing activations by ε and checking if they
        remain aligned with the attractor subspace.
    
    Args:
        model: nnsight model
        layer_idx: Layer where attractor lives
        attractor_subspace: The attractor to test
        target_prompts: Prompts that should be in attractor basin
        perturbation_sizes: List of perturbation magnitudes
        
    Returns:
        List of dicts with strength metrics for each perturbation size
    """
    print(f"Measuring attractor strength with {len(target_prompts)} samples...")
    
    results = []
    
    # Collect baseline activations
    baseline_activations = []
    for prompt in target_prompts:
        try:
            with model.trace(prompt):
                act = model.model.layers[layer_idx].output[0].save()
            val = act.value if hasattr(act, 'value') else act
            x = val[0, -1, :].cpu() if val.ndim == 3 else val[-1, :].cpu()
            baseline_activations.append(x)
        except:
            continue
    
    print(f"Collected {len(baseline_activations)} baseline activations")
    
    # Test each perturbation size
    for epsilon in perturbation_sizes:
        alignments_after = []
        
        for x in baseline_activations:
            # Add Gaussian noise
            noise = torch.randn_like(x) * epsilon
            x_perturbed = x + noise
            
            # Measure alignment after perturbation
            metrics = measure_attractor_distance(x_perturbed, attractor_subspace)
            alignments_after.append(metrics['alignment'])
        
        results.append({
            'perturbation_size': epsilon,
            'mean_alignment': np.mean(alignments_after),
            'std_alignment': np.std(alignments_after),
            'min_alignment': np.min(alignments_after),
            'max_alignment': np.max(alignments_after),
            'n_samples': len(alignments_after)
        })
        
        print(f"  ε={epsilon:.2f}: alignment={np.mean(alignments_after):.3f} ± {np.std(alignments_after):.3f}")
    
    return results


# ==============================================================================
# COMPARATIVE ANALYSIS
# ==============================================================================

def compare_attractors(attractor1, attractor2, name1="Attractor 1", name2="Attractor 2"):
    """
    Compare two attractor subspaces to see if they overlap or are distinct.
    
    Mathematical Framework:
        For subspaces S₁ and S₂, we measure:
        1. Principal angles θᵢ between subspaces
        2. Projection overlap: ||P₁P₂||_F
        3. Basis similarity: max cosine similarity between basis vectors
    
    Args:
        attractor1, attractor2: Attractor dicts from extract_attractor_subspace
        name1, name2: Names for display
        
    Returns:
        dict with comparison metrics
    """
    print(f"\nComparing {name1} vs {name2}:")
    
    P1 = attractor1['projection_matrix']
    P2 = attractor2['projection_matrix']
    B1 = attractor1['basis']  # [d_model, k1]
    B2 = attractor2['basis']  # [d_model, k2]
    
    # Metric 1: Projection overlap (Frobenius norm of composition)
    overlap = torch.norm(P1 @ P2, p='fro') / torch.norm(P1, p='fro')
    print(f"  Projection overlap: {overlap:.3f}")
    
    # Metric 2: Principal angles
    # Compute SVD of B1^T @ B2
    M = B1.T @ B2  # [k1, k2]
    U, S, Vh = torch.linalg.svd(M, full_matrices=False)
    
    # S contains cosines of principal angles
    principal_angles = torch.acos(torch.clamp(S, -1, 1)) * 180 / np.pi
    print(f"  Principal angles (degrees): {principal_angles[:5].detach().cpu().numpy()}")
    print(f"  Min angle: {principal_angles.min():.1f}°")
    print(f"  Max angle: {principal_angles.max():.1f}°")
    
    # Metric 3: Maximum basis similarity
    similarities = B1.T @ B2  # [k1, k2]
    max_sim = torch.max(torch.abs(similarities))
    print(f"  Max basis vector similarity: {max_sim:.3f}")
    
    # Metric 4: Effective dimension comparison
    dim1 = attractor1['effective_dim']
    dim2 = attractor2['effective_dim']
    print(f"  Dimensions: {dim1} vs {dim2}")
    
    # Interpretation
    if overlap > 0.7:
        print(f"  → {name1} and {name2} are SIMILAR (high overlap)")
    elif overlap > 0.3:
        print(f"  → {name1} and {name2} are RELATED (moderate overlap)")
    else:
        print(f"  → {name1} and {name2} are DISTINCT (low overlap)")
    
    return {
        'projection_overlap': overlap.item(),
        'principal_angles': principal_angles.detach().cpu().numpy(),
        'min_angle': principal_angles.min().item(),
        'max_angle': principal_angles.max().item(),
        'max_basis_similarity': max_sim.item(),
        'dim1': dim1,
        'dim2': dim2
    }


# ==============================================================================
# EXAMPLE USAGE
# ==============================================================================

if __name__ == "__main__":
    print("""
Attractor Analysis Tools Loaded!

Example workflow:

# 1. Extract attractor subspace
safety_attractor = extract_attractor_subspace(
    model, hf_model, layer_idx=16, head_idx=2,
    target_prompts=refusal_prompts
)

# 2. Measure alignment for new prompts
alignments = batch_measure_alignment(
    model, test_prompts, layer_idx=16, 
    attractor_subspace=safety_attractor
)

# 3. Trace attractor formation
trajectory = trace_attractor_trajectory(
    model, "How to hack a bank",
    safety_attractor, start_layer=10, end_layer=20
)

# 4. Apply projection steering
text = apply_projection_steering(
    model, layer_idx=16, attractor_subspace=safety_attractor,
    prompt="How to make explosives", steering_strength=2.0, invert=True
)

# 5. Compare attractors
comparison = compare_attractors(
    safety_attractor, bias_attractor,
    name1="Safety", name2="Bias"
)

See projection_theory_analysis_mats.md for detailed documentation!
    """)
