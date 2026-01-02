"""
Spectral MI Extensions for Reasoning Analysis
==============================================

Extension module for analyzing Chain-of-Thought reasoning, algorithmic computation,
and connecting to mechanistic interpretability research (Neel Nanda, etc.)

Add this to your spectral_mi.py or import as separate module.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional

# ==============================================================================
# REASONING ATTRACTOR ANALYSIS
# ==============================================================================

def extract_reasoning_attractor(model, layer_idx: int, 
                               reasoning_prompts: List[str],
                               variance_threshold: float = 0.95) -> Dict:
    """
    Extract the low-dimensional "reasoning manifold" from examples.
    
    Similar to safety attractor extraction, but for algorithmic computation.
    
    Args:
        model: nnsight LanguageModel
        layer_idx: Which layer to analyze
        reasoning_prompts: Examples of correct reasoning (e.g., arithmetic solutions)
        variance_threshold: How much variance to capture (default 95%)
    
    Returns:
        dict with keys:
            - 'basis': [k, d_model] orthonormal basis of attractor
            - 'singular_values': [k] importance of each direction
            - 'mean': [d_model] center of attractor
            - 'k': dimensionality
            - 'explained_variance': fraction of variance captured
            - 'projection_matrix': [d_model, d_model] for projecting onto attractor
    
    Example:
        >>> arithmetic_prompts = ["23+45=68", "7*9=63", "100-37=63"]
        >>> attractor = extract_reasoning_attractor(model, 16, arithmetic_prompts)
        >>> print(f"Arithmetic lives in {attractor['k']} dimensions")
    """
    activations = []
    
    print(f"Extracting reasoning attractor from Layer {layer_idx}...")
    print(f"  Using {len(reasoning_prompts)} examples")
    
    for prompt in reasoning_prompts:
        try:
            with model.trace(prompt, validate=False, scan=False):
                acts = model.model.layers[layer_idx].output[0].save()
            
            # Extract last token (where answer is formed)
            val = acts.value if hasattr(acts, 'value') else acts
            if val.ndim == 3: 
                v = val[0, -1, :]  # [d_model]
            elif val.ndim == 2: 
                v = val[-1] if val.shape[0] > 1 else val[0]
            else:
                print(f"    Warning: Unexpected shape {val.shape}")
                continue
            
            activations.append(v.detach().cpu().float())
            
        except Exception as e:
            print(f"    Error processing '{prompt[:30]}...': {e}")
            continue
    
    if len(activations) < 2:
        print(f"  ✗ Failed: Need at least 2 valid activations, got {len(activations)}")
        return None
    
    print(f"  ✓ Extracted {len(activations)} activations")
    
    # Stack: [n_samples, d_model]
    X = torch.stack(activations)
    
    # Center data
    mean = X.mean(dim=0)
    X_centered = X - mean
    
    # SVD to find principal components
    U, S, Vh = torch.linalg.svd(X_centered, full_matrices=False)
    
    # Determine dimensionality by variance threshold
    variance = (S ** 2) / (S ** 2).sum()
    cumvar = torch.cumsum(variance, dim=0)
    k = int((cumvar < variance_threshold).sum()) + 1
    k = min(k, len(S))  # Can't exceed number of samples
    
    # Extract k-dimensional subspace
    basis = Vh[:k, :]  # [k, d_model]
    singular_values = S[:k]
    
    # Create projection matrix: P = V^T @ V
    projection_matrix = basis.T @ basis  # [d_model, d_model]
    
    print(f"  ✓ Attractor dimension: k={k} out of {X.shape[-1]}")
    print(f"    Explains {cumvar[k-1]*100:.1f}% of variance")
    print(f"    Top 3 singular values: {singular_values[:3].tolist()}")
    
    return {
        'basis': basis,  # [k, d_model]
        'singular_values': singular_values,  # [k]
        'mean': mean,  # [d_model]
        'k': k,
        'explained_variance': cumvar[k-1].item(),
        'projection_matrix': projection_matrix  # [d_model, d_model]
    }


def measure_attractor_alignment(activation: torch.Tensor, 
                               attractor: Dict) -> Dict[str, float]:
    """
    Measure how well an activation aligns with a reasoning attractor.
    
    Returns:
        dict with:
            - 'alignment': fraction of activation in attractor subspace (0-1)
            - 'distance': distance from attractor center
            - 'in_attractor': boolean (alignment > 0.8)
    
    Example:
        >>> test_activation = get_activation(model, "47+89=136", layer=16)
        >>> metrics = measure_attractor_alignment(test_activation, arithmetic_attractor)
        >>> print(f"Alignment: {metrics['alignment']:.3f}")
    """
    if activation.device != attractor['projection_matrix'].device:
        activation = activation.to(attractor['projection_matrix'].device)
    
    # Center
    x_centered = activation - attractor['mean'].to(activation.device)
    
    # Project onto attractor
    P = attractor['projection_matrix'].to(activation.device)
    x_proj = x_centered @ P.T
    
    # Alignment = ||x_proj|| / ||x||
    norm_proj = torch.norm(x_proj)
    norm_total = torch.norm(x_centered)
    
    alignment = (norm_proj / (norm_total + 1e-8)).item()
    
    # Distance from center
    distance = norm_total.item()
    
    return {
        'alignment': alignment,
        'distance': distance,
        'in_attractor': alignment > 0.8
    }


# ==============================================================================
# REASONING TRAJECTORY ANALYSIS
# ==============================================================================

def trace_reasoning_trajectory(model, prompt: str, 
                               layer_idx: int = 16,
                               max_tokens: int = 50,
                               stop_tokens: List[str] = [".", "!", "\n\n"],
                               attractor: Optional[Dict] = None) -> Dict:
    """
    Track spectral rigidity at each token during generation.
    Detects "phase transitions" when model commits to answer.
    
    Args:
        model: nnsight model
        prompt: Starting text
        target_layer: Which layer to monitor
        max_tokens: Maximum generation length
        stop_tokens: When to stop (default: sentence endings)
    
    Returns:
        dict with:
            - 'tokens': List of generated tokens
            - 'rigidities': List of rigidity values
            - 'alignments': List of alignment values (if attractor provided)
            - 'full_text': Complete generated text
            - 'phase_transition': Token index where rigidity jumps (if any)
    
    Example:
        >>> traj = trace_reasoning_trajectory(model, "What is 7*8? Let's calculate:")
        >>> plt.plot(traj['rigidities'])
        >>> plt.xlabel('Token'); plt.ylabel('Rigidity')
    """
    from spectral_mi import calc_relative_entropy
    
    trajectory = []
    alignments = []
    tokens = []
    current_text = prompt
    
    print(f"Tracing reasoning trajectory from layer {layer_idx}...")
    print(f"  Starting: '{prompt[:50]}...'")
    
    for step in range(max_tokens):
        # Generate next token
        try:
            with model.generate(current_text, max_new_tokens=1, remote=False):
                out = model.generator.output.save()
            
            out_val = out.value if hasattr(out, 'value') else out
            if len(out_val) == 0 or len(out_val[0]) == 0:
                break
            
            next_token_id = out_val[0][-1].item()
            next_token = model.tokenizer.decode([next_token_id])
            
            current_text += next_token
            tokens.append(next_token)
            
            # Measure rigidity at this step
            with model.trace(current_text, validate=False, scan=False):
                acts = model.model.layers[layer_idx].output[0].save()
            
            r = calc_relative_entropy(acts)
            trajectory.append(r)
            
            # Measure Alignment
            if attractor is not None:
                val = acts.value if hasattr(acts, 'value') else acts
                if val.ndim == 3: v = val[0, -1, :]
                else: v = val[-1] if val.shape[0] > 1 else val[0]
                
                align = measure_attractor_alignment(v, attractor)
                alignments.append(align['alignment'])
            
            # Check stop condition
            if any(stop_tok in next_token for stop_tok in stop_tokens):
                print(f"  Stopped at token {step+1}: '{next_token}'")
                break
                
            if next_token_id == model.tokenizer.eos_token_id:
                print(f"  Reached EOS at token {step+1}")
                break
                
        except Exception as e:
            print(f"  Error at step {step}: {e}")
            break
    
    # Detect phase transition (sudden rigidity jump)
    phase_transition = None
    if len(trajectory) > 3:
        diffs = np.diff(trajectory)
        # Look for jump > 0.15
        large_jumps = np.where(diffs > 0.15)[0]
        if len(large_jumps) > 0:
            phase_transition = int(large_jumps[0])
            print(f"  ⚡ Phase transition detected at token {phase_transition}")
            print(f"     Rigidity jumped from {trajectory[phase_transition]:.3f} → {trajectory[phase_transition+1]:.3f}")
    
    print(f"  ✓ Generated {len(tokens)} tokens")
    print(f"    Final rigidity: {trajectory[-1]:.3f}")
    
    return {
        'tokens': tokens,
        'rigidities': trajectory,
        'alignments': alignments,
        'rigidities': trajectory,
        'full_text': current_text,
        'phase_transition': phase_transition
    }


# ==============================================================================
# COMPARATIVE ANALYSIS (CoT vs Direct)
# ==============================================================================

def compare_reasoning_modes(model, problem: str,
                           direct_suffix: str = " Answer:",
                           cot_suffix: str = " Let's think step by step:\n",
                           layers_to_scan: range = range(10, 25),
                           max_tokens: int = 30) -> Dict:
    """
    Compare spectral dynamics for direct answer vs. Chain-of-Thought.
    
    Hypothesis: Direct answer locks in early (high rigidity at L10-15)
                CoT allows more exploration (gradual rigidity increase)
    
    Args:
        problem: The question (e.g., "What is 47+89?")
        direct_suffix: Prompt suffix for direct answer
        cot_suffix: Prompt suffix for CoT
        layers_to_scan: Which layers to measure
    
    Returns:
        dict with:
            - 'direct': {'text': str, 'rigidities': [float]}
            - 'cot': {'text': str, 'rigidities': [float]}
            - 'difference': [float] (cot - direct at each layer)
    
    Example:
        >>> results = compare_reasoning_modes(model, "What is 47+89?")
        >>> plt.plot(results['direct']['rigidities'], label='Direct')
        >>> plt.plot(results['cot']['rigidities'], label='CoT')
        >>> plt.legend()
    """
    from spectral_mi import calc_relative_entropy
    
    print(f"Comparing reasoning modes for: '{problem}'")
    
    results = {}
    
    for mode in ['direct', 'cot']:
        suffix = direct_suffix if mode == 'direct' else cot_suffix
        full_prompt = problem + suffix
        
        print(f"\n  [{mode.upper()}] Generating...")
        
        # Generate
        with model.generate(full_prompt, max_new_tokens=max_tokens, remote=False):
            out = model.generator.output.save()
        
        out_val = out.value if hasattr(out, 'value') else out
        text = model.tokenizer.decode(out_val[0])
        print(f"    Generated: '{text[len(full_prompt):].strip()[:50]}...'")
        
        # Scan layers
        layer_rigidities = []
        
        for layer_idx in layers_to_scan:
            with model.trace(text, validate=False, scan=False):
                acts = model.model.layers[layer_idx].output[0].save()
            
            r = calc_relative_entropy(acts)
            layer_rigidities.append(r)
        
        results[mode] = {
            'text': text,
            'rigidities': layer_rigidities,
            'peak_layer': int(np.argmax(layer_rigidities) + layers_to_scan[0]),
            'peak_rigidity': float(np.max(layer_rigidities))
        }
        
        print(f"    Peak: Layer {results[mode]['peak_layer']} (R={results[mode]['peak_rigidity']:.3f})")
    
    # Calculate difference
    difference = [cot - direct for cot, direct in 
                  zip(results['cot']['rigidities'], results['direct']['rigidities'])]
    
    results['difference'] = difference
    results['cot_delayed_peak'] = results['cot']['peak_layer'] > results['direct']['peak_layer']
    
    print(f"\n  ✓ Analysis complete")
    print(f"    Direct peaks at L{results['direct']['peak_layer']}")
    print(f"    CoT peaks at L{results['cot']['peak_layer']}")
    print(f"    CoT delayed: {results['cot_delayed_peak']}")
    
    return results


# ==============================================================================
# REASONING HEAD DISCOVERY
# ==============================================================================

def find_reasoning_enforcer_heads(model, model_hf, 
                                  reasoning_type: str = "arithmetic",
                                  layers_to_scan: range = range(10, 21)) -> Dict:
    """
    Scan for heads with high rigidity during specific reasoning tasks.
    Discovers "enforcer heads" similar to safety/bias enforcers.
    
    Args:
        reasoning_type: One of ["arithmetic", "logical", "pattern", "analogical"]
        layers_to_scan: Which layers to search
    
    Returns:
        dict with:
            - 'peak_layer': int
            - 'peak_head': int
            - 'peak_rigidity': float
            - 'all_scores': {layer_idx: (best_head, rigidity, [all_rigs])}
    
    Example:
        >>> circuit = find_reasoning_enforcer_heads(model, model_hf, "arithmetic")
        >>> print(f"Arithmetic enforcer: L{circuit['peak_layer']}.H{circuit['peak_head']}")
    """
    from spectral_mi import scan_layer_heads
    
    # Define reasoning-specific test prompts
    prompt_sets = {
        "arithmetic": [
            "Calculate: 47 + 89 = 136",
            "What is 23 * 4? Answer: 92",
            "Solve: 156 - 78 = 78",
            "Divide: 144 / 12 = 12"
        ],
        "logic": [
            "If A→B and B→C, then A→C (transitive reasoning)",
            "All X are Y. Z is X. Therefore Z is Y (modus ponens)",
            "If not B, then not A. B is true. Therefore A is true.",
            "P or Q. Not P. Therefore Q (disjunctive syllogism)"
        ],
        "symbolic": [
            "Reverse: 'hello' -> 'olleh'",
            "Format: '2023-10-15' -> '15/10/2023'",
            "Extract: 'user@example.com' -> 'example.com'",
            "Replace: 'cat' with 'dog' in 'the cat sat'"
        ],
        "pattern": [
            "Sequence: 2,4,6,8,... pattern is add 2",
            "Pattern: A,C,E,G,... rule is skip one letter",
            "Series: 1,4,9,16,... is perfect squares",
            "Sequence: 1,1,2,3,5,8,... is Fibonacci"
        ],
        "analogical": [
            "King is to Queen as Prince is to Princess",
            "Hot is to Cold as Day is to Night",
            "Cat is to Meow as Dog is to Bark",
            "Up is to Down as Left is to Right"
        ]
    }
    
    if reasoning_type not in prompt_sets:
        raise ValueError(f"Unknown reasoning type: {reasoning_type}")
    
    test_prompts = prompt_sets[reasoning_type]
    
    print(f"🔍 Searching for {reasoning_type.upper()} enforcer head...")
    print(f"   Scanning layers {layers_to_scan.start}-{layers_to_scan.stop-1}")
    print(f"   Using {len(test_prompts)} test prompts")
    
    all_scores = {}
    
    for layer_idx in layers_to_scan:
        print(f"\n   Layer {layer_idx}:")
        
        layer_results = []
        
        for i, prompt in enumerate(test_prompts):
            best_head, max_rig, all_rigs = scan_layer_heads(
                model, model_hf, layer_idx, prompt, max_new_tokens=20
            )
            layer_results.append((best_head, max_rig))
            print(f"      Prompt {i+1}: Head {best_head} (R={max_rig:.3f})")
        
        # Consensus: most common head + average rigidity
        heads = [h for h, r in layer_results]
        rigs = [r for h, r in layer_results]
        
        # Most frequent head
        from collections import Counter
        head_counts = Counter(heads)
        consensus_head = head_counts.most_common(1)[0][0]
        
        # Average rigidity
        avg_rig = np.mean(rigs)
        
        all_scores[layer_idx] = (consensus_head, avg_rig, rigs)
        
        print(f"      → Consensus: Head {consensus_head} (avg R={avg_rig:.3f})")
    
    # Find peak
    peak_layer = max(all_scores.keys(), key=lambda k: all_scores[k][1])
    peak_head, peak_rig, _ = all_scores[peak_layer]
    
    print(f"\n   🎯 {reasoning_type.upper()} ENFORCER HEAD FOUND!")
    print(f"      Layer {peak_layer}, Head {peak_head}")
    print(f"      Rigidity: {peak_rig:.3f}")
    
    return {
        'peak_layer': peak_layer,
        'peak_head': peak_head,
        'peak_rigidity': peak_rig,
        'all_scores': all_scores,
        'reasoning_type': reasoning_type
    }


# ==============================================================================
# VISUALIZATION TOOLS
# ==============================================================================

def plot_reasoning_trajectory(trajectory_data: Dict, save_path: Optional[str] = None):
    """
    Visualize the rigidity trajectory with token annotations.
    """
    rigidities = trajectory_data['rigidities']
    tokens = trajectory_data['tokens']
    phase_transition = trajectory_data.get('phase_transition')
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot trajectory
    ax.plot(rigidities, marker='o', linewidth=2, markersize=6, color='steelblue')
    
    # Mark phase transition if detected
    if phase_transition is not None:
        ax.axvline(x=phase_transition, color='red', linestyle='--', 
                  linewidth=2, label='Phase Transition')
        ax.scatter([phase_transition], [rigidities[phase_transition]], 
                  color='red', s=200, zorder=5, marker='*')
    
    # Annotate some tokens
    for i in range(0, len(tokens), max(1, len(tokens)//10)):
        token = tokens[i].replace('\n', '\\n')
        ax.annotate(token, (i, rigidities[i]), 
                   xytext=(0, 10), textcoords='offset points',
                   fontsize=8, rotation=45, ha='left')
    
    ax.set_xlabel('Generation Step', fontsize=12)
    ax.set_ylabel('Spectral Rigidity', fontsize=12)
    ax.set_title('Reasoning Trajectory: Token-by-Token Rigidity', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, label='Moderate Rigidity')
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved trajectory plot to {save_path}")
    
    return fig


def plot_cot_vs_direct(comparison_data: Dict, save_path: Optional[str] = None):
    """
    Visualize CoT vs Direct Answer rigidity profiles.
    """
    direct_rigs = comparison_data['direct']['rigidities']
    cot_rigs = comparison_data['cot']['rigidities']
    layers = range(10, 10 + len(direct_rigs))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Comparison
    ax1.plot(layers, direct_rigs, marker='o', linewidth=2.5, 
            label='Direct Answer', color='coral')
    ax1.plot(layers, cot_rigs, marker='s', linewidth=2.5, 
            label='Chain-of-Thought', color='steelblue')
    
    # Mark peaks
    ax1.axvline(x=comparison_data['direct']['peak_layer'], 
               color='coral', linestyle='--', alpha=0.5)
    ax1.axvline(x=comparison_data['cot']['peak_layer'], 
               color='steelblue', linestyle='--', alpha=0.5)
    
    ax1.set_xlabel('Layer', fontsize=12)
    ax1.set_ylabel('Spectral Rigidity', fontsize=12)
    ax1.set_title('Rigidity Profiles', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Right: Difference
    ax2.bar(layers, comparison_data['difference'], 
           color=['green' if d > 0 else 'red' for d in comparison_data['difference']],
           alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.set_xlabel('Layer', fontsize=12)
    ax2.set_ylabel('CoT - Direct', fontsize=12)
    ax2.set_title('Rigidity Difference (CoT - Direct)', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved comparison plot to {save_path}")
    
    return fig


# ==============================================================================
# QUICK ANALYSIS FUNCTION
# ==============================================================================

def quick_reasoning_analysis(model, model_hf, problem: str, 
                             target_layer: int = 16) -> Dict:
    """
    Run a quick comprehensive analysis of reasoning for a problem.
    
    Args:
        problem: The question/problem to analyze
        target_layer: Primary layer to focus on (default 16)
    
    Returns:
        dict with all analysis results
    
    Example:
        >>> results = quick_reasoning_analysis(model, model_hf, "What is 47+89?")
        >>> print(results['summary'])
    """
    print("="*70)
    print("QUICK REASONING ANALYSIS")
    print("="*70)
    print(f"Problem: {problem}")
    print(f"Target Layer: {target_layer}")
    print()
    
    # 1. Compare CoT vs Direct
    print("📊 Step 1: Comparing CoT vs Direct Answer...")
    comparison = compare_reasoning_modes(model, problem)
    
    # 2. Trace CoT trajectory
    print("\n📈 Step 2: Tracing CoT Reasoning Trajectory...")
    cot_prompt = problem + " Let's think step by step:\n"
    trajectory = trace_reasoning_trajectory(model, cot_prompt, target_layer=target_layer)
    
    # 3. Summary
    summary = f"""
    ANALYSIS SUMMARY
    ================
    Problem: {problem}
    
    Direct Answer:
      - Peak Layer: {comparison['direct']['peak_layer']}
      - Peak Rigidity: {comparison['direct']['peak_rigidity']:.3f}
      - Output: {comparison['direct']['text'][-50:]}
    
    Chain-of-Thought:
      - Peak Layer: {comparison['cot']['peak_layer']}
      - Peak Rigidity: {comparison['cot']['peak_rigidity']:.3f}
      - Phase Transition: {'Yes, at token ' + str(trajectory['phase_transition']) if trajectory['phase_transition'] else 'Not detected'}
      - Final Rigidity: {trajectory['rigidities'][-1]:.3f}
      - Output: {comparison['cot']['text'][-50:]}
    
    Key Findings:
      - CoT delayed peak: {comparison['cot_delayed_peak']}
      - CoT explores more: {comparison['cot']['peak_layer'] > comparison['direct']['peak_layer']}
    """
    
    print(summary)
    
    return {
        'comparison': comparison,
        'trajectory': trajectory,
        'summary': summary
    }


if __name__ == "__main__":
    print("Spectral MI Reasoning Extensions loaded!")
    print("\nAvailable functions:")
    print("  - extract_reasoning_attractor()")
    print("  - trace_reasoning_trajectory()")
    print("  - compare_reasoning_modes()")
    print("  - find_reasoning_enforcer_heads()")
    print("  - quick_reasoning_analysis()")
